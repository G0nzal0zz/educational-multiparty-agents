import asyncio

import numpy as np
import whisper

from src.lib.events import STTOutputEvent

whisperSTT = whisper.load_model("tiny")


class WhisperSTT:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    async def transcribe(self, audio: np.ndarray):
        result = await asyncio.to_thread(
            whisperSTT.transcribe,
            audio,
            language="en",
        )
        text = result["text"]

        if text:
            yield STTOutputEvent.create(text)
