import asyncio
import queue
import threading
import time
from typing import Literal

import sounddevice as sd
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from shared_lib.events import Role

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

    audio_player_stop: bool

    def __init__(
        self, tts_queue: asyncio.Queue[TTSEndEvent], text_queue: queue.Queue[str | None]
    ):
        self.tts_queue = tts_queue
        self.text_queue = text_queue

        self.model = ChatterboxTurboTTS.from_pretrained(device=device)

        self.audio_thread = None
        self.audio_player_stop = False

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
                if audio_chunk is None or self.audio_player_stop:  # Sentinel to stop
                    break
                self.play_audio_chunk(audio_chunk, sample_rate)
                audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio player error: {e}")

    def stop_audio_player(self):
        if self.audio_thread is None or self.audio_thread.is_alive() is False:
            print("Can't stop audio player since it isn't running")
            return

        print("Stopping audio player")
        self.audio_player_stop = True

    def start(
        self, role: Literal[Role.TEACHER, Role.STUDENT], loop: asyncio.AbstractEventLoop
    ):
        # Setup audio playback queue and thread
        audio_queue = queue.Queue()
        self.audio_thread = threading.Thread(
            target=self.audio_player_worker, args=(audio_queue, self.model.sr)
        )
        self.audio_thread.daemon = True
        self.audio_thread.start()
        self.audio_player_stop = False

        while True:
            try:
                # If there is already audio in the queue, wait until playback starts
                # to avoid overloading the GPU and potentially running out of memory
                if audio_queue.qsize() >= AUDIO_QUEUE_MAX_WAIT:
                    time.sleep(1)
                    continue
                text_chunk = self.text_queue.get()

                if text_chunk is None:
                    break

                initial_time = time.time()
                for audio_chunk in self.model.generate(
                    text=text_chunk,
                    exaggeration=0.5,
                    temperature=0.8,
                    cfg_weight=0.5,
                ):
                    print(f"Chunk time = {time.time() - initial_time}")

                    # Queue audio for immediate playback
                    audio_queue.put(audio_chunk.clone())

            except Exception as e:
                print(f"Error during streaming generation: {e}")

        print("Chatterbox finished generating audio")
        audio_queue.join()  # Wait for all audio to finish playing
        audio_queue.put(None)  # Sentinel to stop thread

        print("Audio finished playing")
        self.audio_thread = None
        _ = asyncio.run_coroutine_threadsafe(
            self.tts_queue.put(TTSEndEvent.create(role)), loop
        )
