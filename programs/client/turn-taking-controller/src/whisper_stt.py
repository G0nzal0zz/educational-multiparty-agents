import argparse
import asyncio
import os
from datetime import datetime, timedelta
from queue import Queue
from sys import platform
from time import sleep

import numpy as np
import shared_lib.events
import speech_recognition as sr
import torch
import whisper


class WhisperSTT:
    recorder: sr.Recognizer 

    source: sr.Microphone

    record_timeout: int

    phrase_timeout: int

    audio_model: whisper.Whisper

    def __init__(self, args: argparse.Namespace, source):
        # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = args.energy_threshold
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
        self.recorder.dynamic_energy_threshold = False

        self.source = source

        self.record_timeout = args.record_timeout
        self.phrase_timeout = args.phrase_timeout

        # Load / Download mode
        model = args.model
        if args.model != "large" and not args.non_english:
            model = model + ".en"
        self.audio_model = whisper.load_model(model)

    async def STT(self):
        phrase_time = None
        phrase_bytes = bytes()
        transcription = [""]

        data_queue = asyncio.Queue()

        loop = asyncio.get_running_loop()

        def record_callback(_, audio):
            data = audio.get_raw_data()
            loop.call_soon_threadsafe(data_queue.put_nowait, data)

        self.recorder.listen_in_background(
            self.source,
            record_callback,
            phrase_time_limit=self.record_timeout,
        )

        while True:
            now = datetime.utcnow()

            try:
                data = await data_queue.get()

                phrase_complete = False

                if phrase_time and now - phrase_time > timedelta(
                    seconds=self.phrase_timeout
                ):
                    phrase_bytes = bytes()
                    phrase_complete = True

                phrase_time = now
                phrase_bytes += data

                audio_np = (
                    np.frombuffer(phrase_bytes, dtype=np.int16)
                    .astype(np.float32) / 32768.0
                )

                result = await asyncio.to_thread(
                    self.audio_model.transcribe,
                    audio_np,
                    fp16=torch.cuda.is_available(),
                )

                text = result["text"].strip()

                if phrase_complete and isinstance(text, str):
                    transcription.append(text)
                    yield shared_lib.events.UserInputEvent.create(text)
                else:
                    transcription[-1] = text

            except asyncio.CancelledError:
                break
