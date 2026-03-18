import asyncio
import queue
import threading
import time
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Literal

import numpy as np
import sounddevice as sd
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS
from shared_lib.events import Role, SocketAgentTextChunkEvent

from events import TTSEndEvent

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

AUDIO_QUEUE_MAX_WAIT = 1
"""Specifies the maximum number of items allowed in the audio queue
before processing a new agent text chunk into audio."""


class ChatterboxTTS:
    tts_queue: asyncio.Queue[TTSEndEvent]
    text_queue: queue.Queue[str | None]
    model: ChatterboxTurboTTS

    def __init__(
        self, tts_queue: asyncio.Queue[TTSEndEvent], text_queue: queue.Queue[str | None]
    ):
        self.tts_queue = tts_queue
        self.text_queue = text_queue

        self.model = ChatterboxTurboTTS.from_pretrained(device=device)

    def play_audio_chunk(self, audio_chunk, sample_rate):
        """Play audio chunk using sounddevice with proper sequencing"""
        try:
            # Convert to numpy and play with sounddevice
            audio_np = audio_chunk.squeeze().numpy()
            sd.play(audio_np, sample_rate)
            sd.wait()  # Wait for this chunk to finish before returning
        except Exception as e:
            print(f"Error playing audio: {e}")

    def audio_player_worker(self, audio_queue, sample_rate):
        """Worker thread that plays audio chunks from queue"""
        while True:
            try:
                audio_chunk = audio_queue.get(timeout=1.0)
                if audio_chunk is None:  # Sentinel to stop
                    break
                self.play_audio_chunk(audio_chunk, sample_rate)
                audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio player error: {e}")

    def start(
        self, role: Literal[Role.TEACHER, Role.STUDENT], loop: asyncio.AbstractEventLoop
    ):
        chunk_count = 0

        # Setup audio playback queue and thread
        audio_queue = queue.Queue()
        audio_thread = threading.Thread(
            target=self.audio_player_worker, args=(audio_queue, self.model.sr)
        )
        audio_thread.daemon = True
        audio_thread.start()

        while True:
            try:
                # If there is already audio in the queue, wait until playback starts
                # to avoid overloading the GPU and potentially running out of memory
                if audio_queue.qsize() >= AUDIO_QUEUE_MAX_WAIT:
                    time.sleep(1)
                    continue
                text_chunk = self.text_queue.get()

                if text_chunk is None:
                    print("STOPING CHATTERBOX LOOP")
                    break

                initial_time = time.time()
                for audio_chunk in self.model.generate(
                    text=text_chunk,
                    exaggeration=0.5,
                    temperature=0.8,
                    cfg_weight=0.5,
                ):
                    print(f"Chunk time = {time.time() - initial_time}")
                    chunk_count += 1

                    # Queue audio for immediate playback
                    audio_queue.put(audio_chunk.clone())

                    # chunk_duration = audio_chunk.shape[-1] / model.sr
                    # print(
                    #     f"Received chunk {chunk_count}, shape: {audio_chunk.shape}, duration: {chunk_duration:.3f}s"
                    # )

                    if chunk_count == 1:
                        print("Audio playback started!")

            except KeyboardInterrupt:
                print("\nPlayback interrupted by user")
            except Exception as e:
                print(f"Error during streaming generation: {e}")
                import traceback

                traceback.print_exc()

        print("Chatterbox finished generating audio")
        audio_queue.join()  # Wait for all audio to finish playing
        audio_queue.put(None)  # Sentinel to stop thread
        print("Audio finished playing")
        _ = asyncio.run_coroutine_threadsafe(
            self.tts_queue.put(TTSEndEvent.create(role)), loop
        )
