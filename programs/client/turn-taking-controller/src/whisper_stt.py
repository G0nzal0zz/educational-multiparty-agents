import argparse
import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import numpy as np
import speech_recognition as sr
import torch
import whisper

from events import STTChunkEvent, STTEndEvent, STTEvent


class WhisperSTT:
    recorder: sr.Recognizer

    source: sr.Microphone

    record_timeout: int

    intervention_timeout: int

    audio_model: whisper.Whisper

    def __init__(self, args: argparse.Namespace, source):
        print("class source", source)
        # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = args.energy_threshold
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
        self.recorder.dynamic_energy_threshold = False

        self.source = source

        self.record_timeout = args.record_timeout
        self.intervention_timeout = args.intervention_timeout

        # Load / Download mode
        model = args.model
        if args.model != "large" and not args.non_english:
            model = model + ".en"
        self.audio_model = whisper.load_model(model)

    async def STT(self) -> AsyncIterator[STTEvent]:
        """
        Transcribe audio from the local microphone.

        Yields transcribed text whenever the user finishes speaking
        a intervention (determined by silence timeout).
        """
        intervention_time = None
        intervention_bytes = bytes()

        data_queue = asyncio.Queue()

        loop = asyncio.get_running_loop()

        with self.source as source:
            self.recorder.adjust_for_ambient_noise(source)

        def record_callback(_, audio):
            data = audio.get_raw_data()
            loop.call_soon_threadsafe(data_queue.put_nowait, data)

        self.recorder.listen_in_background(
            self.source,
            record_callback,
            intervention_time_limit=self.record_timeout,
        )

        transcription = [""]
        while True:
            now = datetime.utcnow()

            try:
                data = await data_queue.get()

                intervention_complete = False

                if intervention_time and now - intervention_time > timedelta(
                    seconds=self.intervention_timeout
                ):
                    intervention_bytes = bytes()
                    intervention_complete = True

                intervention_time = now
                intervention_bytes += data

                audio_np = (
                    np.frombuffer(intervention_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                result = await asyncio.to_thread(
                    self.audio_model.transcribe,
                    audio_np,
                    fp16=torch.cuda.is_available(),
                )

                text = str(result["text"]).strip()
                print("STT text: ", text)

                transcription.append(text)
                if intervention_completed and text:
                    yield STTEndEvent.create(" ".join(map(str, transcription)))
                    transcription = [""]
                else:
                    yield STTChunkEvent.create(text)

            except asyncio.CancelledError:
                break
