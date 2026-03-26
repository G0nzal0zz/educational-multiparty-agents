import argparse
import asyncio
from collections.abc import AsyncIterator

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
        Stream speech-to-text (STT) events from the microphone.

        Pipeline:
            Microphone → raw audio bytes → transcription chunks →
            silence detection → final utterance events

        Yields:
            STTChunkEvent: partial transcription updates
            STTEndEvent: final text after user stops speaking
        """

        # Queues for each stage of the pipeline
        data_queue: asyncio.Queue[bytes] = asyncio.Queue()
        chunk_queue: asyncio.Queue[str] = asyncio.Queue()
        output_queue: asyncio.Queue[STTEvent] = asyncio.Queue()

        loop = asyncio.get_running_loop()

        # --- Microphone setup ---
        with self.source as source:
            # Calibrate mic to ignore ambient noise
            self.recorder.adjust_for_ambient_noise(source)

        def record_callback(_, audio: sr.AudioData):
            """
            Runs in a background thread.
            Pushes raw audio bytes into the async pipeline.
            """
            data = audio.get_raw_data()
            _ = loop.call_soon_threadsafe(data_queue.put_nowait, data)

        # Start background recording
        _ = self.recorder.listen_in_background(
            self.source,
            record_callback,
            phrase_time_limit=self.record_timeout,
        )

        # --- Stage 1: Audio → transcription chunks ---
        async def transcribe_audio():
            """
            Consumes raw audio bytes and produces incremental transcription chunks.
            """
            buffer = bytes()
            last_chunk_len = 0

            try:
                while True:
                    # Accumulate audio data
                    buffer += await data_queue.get()

                    # Convert PCM bytes → normalized float32 numpy array
                    audio_np = (
                        np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )

                    # Run transcription in a worker thread (CPU/GPU bound)
                    result = await asyncio.to_thread(
                        self.audio_model.transcribe,
                        audio_np,
                        fp16=torch.cuda.is_available(),
                    )

                    text = str(result["text"]).strip()
                    if not text or not any(c.isalnum() for c in text):
                        continue

                    chunk = text[last_chunk_len:]
                    last_chunk_len = len(text)

                    # Emit partial chunk
                    await chunk_queue.put(chunk)
                    print(f"[STT] Chunk created: {chunk}")
                    await output_queue.put(STTChunkEvent.create(chunk))

            except asyncio.CancelledError:
                # Graceful shutdown
                return

        # --- Stage 2: Chunk aggregation → end-of-utterance detection ---
        async def detect_utterance_end():
            """
            Groups chunks into a full utterance.
            Emits when silence (timeout) is detected.
            """
            buffer = ""

            try:
                while True:
                    try:
                        # Wait for next chunk, but timeout = silence
                        chunk = await asyncio.wait_for(chunk_queue.get(), timeout=2.0)
                        buffer += chunk

                    except asyncio.TimeoutError:
                        # Silence detected → emit final result
                        if buffer:
                            print(f"[STT] Speech was detected: {buffer}")
                            await output_queue.put(STTEndEvent.create(buffer))
                            buffer = ""
                        # TODO: Remove this else condition
                        else:
                            print("[STT] No speech was detected")

            except asyncio.CancelledError:
                return

        # --- Run pipeline tasks ---
        transcribe_task = asyncio.create_task(transcribe_audio())
        end_task = asyncio.create_task(detect_utterance_end())

        # --- Output stream ---
        try:
            while True:
                # Yield events as they become available
                yield await output_queue.get()

        finally:
            # Ensure background tasks are stopped
            _ = transcribe_task.cancel()
            _ = end_task.cancel()
