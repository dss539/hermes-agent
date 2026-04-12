from __future__ import annotations

"""OpenAI Realtime voice helpers for Discord voice channels.

Fast-path implementation for local/server-side voice conversations:
- authenticates with the existing Hermes-stored OpenAI Codex OAuth token
- opens a Realtime websocket session to ``gpt-realtime``
- accepts Discord PCM utterances (48kHz stereo s16le)
- converts them to Realtime input PCM (24kHz mono s16le)
- collects returned audio deltas and optional transcripts

This module is intentionally narrow in scope: one utterance in, one spoken
reply out, while reusing a persistent websocket session when possible.
"""

import asyncio
import audioop
import base64
import json
import logging
import os
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from hermes_cli.auth import (
    _codex_access_token_is_expiring,
    _read_codex_tokens,
    _refresh_codex_auth_tokens,
)

logger = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def discord_pcm_to_realtime_input(pcm_data: bytes) -> bytes:
    """Convert Discord PCM (48kHz stereo s16le) to Realtime input PCM.

    OpenAI Realtime examples commonly use 24kHz PCM input for websocket audio.
    Discord voice decode gives us 48kHz stereo signed 16-bit PCM.
    """

    if not pcm_data:
        return b""
    # Stereo -> mono, 16-bit samples.
    mono = audioop.tomono(pcm_data, 2, 0.5, 0.5)
    converted, _state = audioop.ratecv(mono, 2, 1, 48000, 24000, None)
    return converted


def write_pcm_wav(path: str | Path, pcm_data: bytes, *, rate: int = 24000, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)


class OpenAIRealtimeVoiceSession:
    URL = "wss://api.openai.com/v1/realtime?model={model}"
    INPUT_RATE = 24000
    OUTPUT_RATE = 24000
    CHUNK_MS = 100

    def __init__(
        self,
        *,
        model: str = "gpt-realtime",
        voice: str = "marin",
        instructions: Optional[str] = None,
    ) -> None:
        self.model = model
        self.voice = voice
        self.instructions = instructions or (
            "You are Hermes in a live voice conversation. Speak naturally, briefly, and "
            "conversationally. Prefer short spoken answers unless asked for more detail. "
            "If interrupted, yield immediately."
        )
        self._ws = None
        self._connect_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._configured = False

    @staticmethod
    def _get_access_token() -> str:
        state = _read_codex_tokens()
        tokens = dict(state.get("tokens") or {})
        access_token = str(tokens.get("access_token") or "")
        if _codex_access_token_is_expiring(access_token, 60):
            tokens = _refresh_codex_auth_tokens(tokens, 20.0)
            access_token = str(tokens.get("access_token") or "")
        if not access_token:
            raise RuntimeError("Missing Codex OAuth access token for OpenAI Realtime")
        return access_token

    async def _recv_json(self, *, timeout: float = 30.0) -> Dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("Realtime websocket is not connected")
        raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        return json.loads(raw)

    async def _send_json(self, payload: Dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime websocket is not connected")
        await self._ws.send(json.dumps(payload))

    async def ensure_connected(self) -> None:
        if self._ws is not None and not getattr(self._ws, "closed", False):
            return
        async with self._connect_lock:
            if self._ws is not None and not getattr(self._ws, "closed", False):
                return
            token = await asyncio.to_thread(self._get_access_token)
            self._ws = await websockets.connect(
                self.URL.format(model=self.model),
                additional_headers={"Authorization": f"Bearer {token}"},
                open_timeout=20,
                close_timeout=5,
                max_size=10 * 1024 * 1024,
            )
            created = await self._recv_json(timeout=15)
            if created.get("type") != "session.created":
                raise RuntimeError(f"Unexpected Realtime hello event: {created.get('type')}")
            await self._send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "model": self.model,
                        "instructions": self.instructions,
                        "output_modalities": ["audio"],
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": self.INPUT_RATE},
                                "turn_detection": None,
                            },
                            "output": {
                                "format": {"type": "audio/pcm", "rate": self.OUTPUT_RATE},
                                "voice": self.voice,
                            },
                        },
                    },
                }
            )
            # session.updated is expected but not strictly required for successful use.
            try:
                evt = await self._recv_json(timeout=10)
                if evt.get("type") == "session.updated":
                    self._configured = True
                elif evt.get("type") == "error":
                    raise RuntimeError(evt.get("error", {}).get("message") or "Realtime session.update failed")
                else:
                    # Some server event orderings may send something else first; remain usable.
                    self._configured = True
            except asyncio.TimeoutError:
                self._configured = True

    async def close(self) -> None:
        ws = self._ws
        self._ws = None
        self._configured = False
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def process_turn(self, discord_pcm: bytes) -> Dict[str, str]:
        """Send one Discord utterance to Realtime and collect the spoken reply."""
        async with self._turn_lock:
            await self.ensure_connected()
            realtime_pcm = await asyncio.to_thread(discord_pcm_to_realtime_input, discord_pcm)
            if not realtime_pcm:
                return {"audio_path": "", "input_transcript": "", "output_transcript": ""}

            bytes_per_chunk = int(self.INPUT_RATE * 2 * (self.CHUNK_MS / 1000.0))
            try:
                await self._send_json({"type": "input_audio_buffer.clear"})
            except Exception:
                # Clearing is helpful but not mandatory.
                pass

            for idx in range(0, len(realtime_pcm), bytes_per_chunk):
                chunk = realtime_pcm[idx : idx + bytes_per_chunk]
                if not chunk:
                    continue
                await self._send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            await self._send_json({"type": "input_audio_buffer.commit"})
            await self._send_json(
                {
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["audio"],
                    },
                }
            )

            audio_bytes = bytearray()
            input_transcript_parts = []
            output_transcript_parts = []

            while True:
                try:
                    event = await self._recv_json(timeout=45)
                except ConnectionClosed:
                    await self.close()
                    raise RuntimeError("Realtime websocket closed during voice turn")

                etype = event.get("type")
                if etype == "response.audio.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        audio_bytes.extend(base64.b64decode(delta))
                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript")
                    if isinstance(transcript, str) and transcript:
                        input_transcript_parts.append(transcript)
                elif etype == "response.audio_transcript.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        output_transcript_parts.append(delta)
                elif etype == "response.audio_transcript.done":
                    transcript = event.get("transcript")
                    if isinstance(transcript, str) and transcript:
                        output_transcript_parts = [transcript]
                elif etype == "response.done":
                    break
                elif etype == "error":
                    err = event.get("error") or {}
                    raise RuntimeError(err.get("message") or json.dumps(err) or "Realtime API error")

            audio_path = ""
            if audio_bytes:
                tmp = NamedTemporaryFile(suffix=".wav", prefix="discord_realtime_reply_", delete=False)
                tmp.close()
                audio_path = tmp.name
                await asyncio.to_thread(write_pcm_wav, audio_path, bytes(audio_bytes), rate=self.OUTPUT_RATE, channels=1)

            return {
                "audio_path": audio_path,
                "input_transcript": "".join(input_transcript_parts).strip(),
                "output_transcript": "".join(output_transcript_parts).strip(),
            }
