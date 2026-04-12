Discord Realtime Voice handoff

Goal
- Make Hermes Discord voice channels use OpenAI Realtime with existing OpenAI Codex OAuth, instead of the old STT -> agent -> TTS path.

What is already done
1. New helper module added:
- gateway/realtime_voice.py
- Responsibilities:
  - reads/refreshes Codex OAuth token via hermes_cli.auth
  - connects to wss://api.openai.com/v1/realtime?model=gpt-realtime
  - configures session.update for audio input/output
  - converts Discord PCM (48kHz stereo s16le) -> 24kHz mono PCM
  - sends input_audio_buffer.append / commit / response.create
  - collects response.audio.delta chunks
  - writes reply audio to temp wav for Discord playback

2. Discord adapter wiring added:
- gateway/platforms/discord.py
- Added:
  - self._voice_audio_callback
  - self._voice_realtime_sessions
  - _process_voice_input short-circuits to _voice_audio_callback if present
  - handle_realtime_voice_input(...) which reuses OpenAIRealtimeVoiceSession and plays audio back in VC
  - leave_voice_channel closes cached realtime session

3. Gateway runner wiring added:
- gateway/run.py
- In _handle_voice_channel_join:
  - if DISCORD_VOICE_REALTIME is enabled, set:
    - adapter._voice_audio_callback = adapter.handle_realtime_voice_input
    - adapter._voice_input_callback = None
  - else keep classic transcript callback path
- join failure / leave clear both callback slots

4. Gateway config bridge added:
- gateway/config.py
- Discord YAML keys now bridge into env vars:
  - discord.voice_realtime -> DISCORD_VOICE_REALTIME
  - discord.realtime_model -> OPENAI_REALTIME_MODEL
  - discord.realtime_voice -> OPENAI_REALTIME_VOICE
  - discord.realtime_instructions -> OPENAI_REALTIME_INSTRUCTIONS

5. Local user config changed:
- ~/.hermes/config.yaml
- under discord:
  - voice_realtime: true
  - realtime_model: gpt-realtime
  - realtime_voice: marin
  - realtime_instructions: You are Hermes in a live voice conversation. Speak naturally, briefly, and conversationally. Prefer short spoken answers unless asked for more detail. If interrupted, yield immediately.

6. Tests added/updated:
- tests/gateway/test_voice_command.py
- Added focused tests for:
  - join wires realtime audio callback when enabled
  - _process_voice_input routes to audio callback when present
  - handle_realtime_voice_input plays reply and cleans temp file
  - leave cleanup still works

Important validated facts
1. Codex OAuth works for Realtime here
- POST /v1/realtime/client_secrets returned 200 with the Codex OAuth bearer
- More importantly, direct websocket auth worked:
  - Authorization: Bearer <codex oauth access token>
  - wss://api.openai.com/v1/realtime?model=gpt-realtime
- So the bridge uses direct bearer auth and does NOT need ephemeral client secrets

2. Realtime session.update requirement discovered
- OpenAI Realtime rejected session.update until this field was included:
  - session.audio.output.format.rate
- Current working shape in gateway/realtime_voice.py uses:
  - audio.input.format.type=audio/pcm
  - audio.input.format.rate=24000
  - audio.output.format.type=audio/pcm
  - audio.output.format.rate=24000
  - audio.output.voice=marin

3. Live smoke passed
- This worked:
  - instantiate OpenAIRealtimeVoiceSession
  - await ensure_connected()
  - session connected and configured true
  - await close()

Verification already run
1. Syntax
- source venv/bin/activate && python -m py_compile gateway/config.py gateway/realtime_voice.py gateway/platforms/discord.py gateway/run.py tests/gateway/test_voice_command.py
- passed

2. Focused tests
- source venv/bin/activate && pytest -q -n0 \
  tests/gateway/test_voice_command.py::TestVoiceChannelCommands::test_join_success_wires_realtime_audio_callback_when_enabled \
  tests/gateway/test_voice_command.py::TestDiscordVoiceChannelMethods::test_process_voice_input_routes_to_audio_callback_when_present \
  tests/gateway/test_voice_command.py::TestDiscordVoiceChannelMethods::test_handle_realtime_voice_input_plays_reply_and_cleans_tempfile \
  tests/gateway/test_voice_command.py::TestDiscordVoiceChannelMethods::test_leave_voice_channel_cleans_up \
  tests/gateway/test_voice_command.py::TestVoiceChannelCommands::test_join_success \
  tests/gateway/test_voice_command.py::TestVoiceChannelCommands::test_leave_success \
  tests/gateway/test_voice_command.py::TestVoiceChannelCommands::test_input_creates_event_and_dispatches
- result: 7 passed

3. Config bridge smoke
- load_gateway_config() correctly set:
  - DISCORD_VOICE_REALTIME=true
  - OPENAI_REALTIME_MODEL=gpt-realtime
  - OPENAI_REALTIME_VOICE=marin
  - OPENAI_REALTIME_INSTRUCTIONS=...

Current blocker
- End-to-end Discord live validation is blocked by missing DISCORD_BOT_TOKEN in this environment.
- scripts/discord-voice-doctor.py reports:
  - DISCORD_BOT_TOKEN not set
- Everything else required for Discord voice looks installed/healthy:
  - discord.py
  - PyNaCl
  - davey
  - opus
  - ffmpeg

Next step to resume
1. Get DISCORD_BOT_TOKEN available
- likely by adding it to ~/.hermes/.env
- after that run:
  - source venv/bin/activate && python scripts/discord-voice-doctor.py
- expect green bot token + permission checks

2. Start the gateway
- likely command to try first:
  - source venv/bin/activate && python -m gateway.run
- or use the hermes gateway entrypoint if preferred in this environment

3. Live test flow
- have user join a Discord server voice channel where the bot is invited
- run /voice channel or equivalent command in the mapped text channel
- confirm bot joins VC
- speak into the channel
- verify:
  - no classic transcript path is used when realtime is enabled
  - reply audio plays in the voice channel
  - leave works and cleans up

4. If live audio fails, inspect these first
- gateway logs around _handle_voice_channel_join
- gateway logs around DiscordAdapter.handle_realtime_voice_input
- any OpenAI Realtime event/error returned from gateway/realtime_voice.py
- ensure ffmpeg playback path in play_in_voice_channel still accepts the generated wav

Known caveats
- gateway/realtime_voice.py currently uses audioop for resampling/downmixing
- audioop is deprecated in Python 3.13+, but works now on this machine
- v1 is a minimal per-utterance speech-to-speech bridge, not a full duplex custom voice framework
- multi-user room behavior is intentionally not blocked

Important warning from implementation process
- gateway/platforms/discord.py is large, and patch-style editing can corrupt or truncate it.
- If it ever gets truncated again, safest recovery is:
  - restore from git show HEAD~1:gateway/platforms/discord.py > gateway/platforms/discord.py
  - or origin/main if needed
  - then reapply targeted edits with a small Python transformation script, not repeated patch hunks

Files changed for this work
- gateway/realtime_voice.py
- gateway/platforms/discord.py
- gateway/run.py
- gateway/config.py
- tests/gateway/test_voice_command.py
- ~/.hermes/config.yaml

Git status before stepping away
- repo has uncommitted changes in the files above
- no commit created yet in this pass

Best resume command set
- cd /home/smithers/.hermes/hermes-agent
- source venv/bin/activate
- python -m py_compile gateway/config.py gateway/realtime_voice.py gateway/platforms/discord.py gateway/run.py tests/gateway/test_voice_command.py
- pytest -q -n0 tests/gateway/test_voice_command.py::TestVoiceChannelCommands::test_join_success_wires_realtime_audio_callback_when_enabled tests/gateway/test_voice_command.py::TestDiscordVoiceChannelMethods::test_process_voice_input_routes_to_audio_callback_when_present tests/gateway/test_voice_command.py::TestDiscordVoiceChannelMethods::test_handle_realtime_voice_input_plays_reply_and_cleans_tempfile tests/gateway/test_voice_command.py::TestDiscordVoiceChannelMethods::test_leave_voice_channel_cleans_up
- python scripts/discord-voice-doctor.py

State summary in one sentence
- The code-side OpenAI Realtime Discord voice bridge is implemented and locally verified; the only remaining blocker for full end-to-end bring-up is providing a Discord bot token.