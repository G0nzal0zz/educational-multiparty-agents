import asyncio
import os
import queue
import random
import threading
import time
import traceback
from concurrent.futures import CancelledError
from pathlib import Path

import sounddevice as sd
import torch
import torchaudio
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals
from huggingface_hub import snapshot_download
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
REPO_ID = "ResembleAI/chatterbox-turbo"
SAMPLES_DIR = "./voices/samples"


class ChatterboxTTS:
    input_event_queue: asyncio.Queue[SocketAgentTextChunkEvent | None]
    model: ChatterboxTurboTTS

    _output_queue: queue.Queue
    _audio_queue: queue.Queue
    _loop: asyncio.AbstractEventLoop | None
    _worker_thread: threading.Thread | None
    _tts_end_event: threading.Event

    def __init__(
        self, input_event_queue: asyncio.Queue[SocketAgentTextChunkEvent | None]
    ):
        self.input_event_queue = input_event_queue

        self.model = ChatterboxTurboTTS.from_pretrained(device=device)

        self.audio_thread = None
        self._output_queue = queue.Queue()
        self._audio_queue = queue.Queue()
        self._loop = None
        self._worker_thread = None
        self._generate_task: asyncio.Task | None = None
        self._tts_end_event = threading.Event()

    def play_audio_chunk(self, audio_chunk, sample_rate):
        """Play audio chunk using sounddevice with proper sequencing"""
        try:
            # Convert to numpy and play with sounddevice
            audio_np = audio_chunk.squeeze().numpy()
            sd.play(audio_np, sample_rate)

        except Exception as e:
            print(f"ERROR: Playing audio: {e}")

    def play_audio_sample(self):
        """Pick a random audio sample from voices/samples/ and queue it"""
        try:
            if not os.path.exists(SAMPLES_DIR):
                print(f"[ERROR] Samples directory does not exist: {SAMPLES_DIR}")
                return

            sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(".wav")]
            if not sample_files:
                print("[ERROR] No sample files found in voices/samples/")
                return

            sample_file = os.path.join(SAMPLES_DIR, random.choice(sample_files))
            audio_sample, sr = torchaudio.load(sample_file)
            self._audio_queue.put(audio_sample)
        except Exception as e:
            print(f"[ERROR] Playing audio sample: {e}")

    def _is_audio_playing(self):
        try:
            return sd.get_stream().active
        except Exception:
            return False

    def audio_player_worker(self, audio_queue, sample_rate):
        """Worker thread that plays audio chunks from queue"""
        while True:
            try:
                time.sleep(0.5)
                if self._is_audio_playing():
                    continue
                audio_chunk = audio_queue.get()
                if audio_chunk is None:
                    self._tts_end_event.set()
                    continue
                self.play_audio_chunk(audio_chunk, sample_rate)
                audio_queue.task_done()
            except Exception as e:
                print(f"ERROR: Audio player worker: {e}")

    def _generate_and_queue(self, text: str, audio_path: str | None):
        """Sync wrapper that runs model.generate and puts chunks in queue"""
        try:
            for audio_chunk in self.model.generate(
                text=text,
                audio_prompt_path=audio_path,
                temperature=0.8,
                cfg_weight=0.5,
            ):
                self._audio_queue.put(audio_chunk.clone())
        except asyncio.CancelledError:
            if not self.input_event_queue.empty():
                self.input_event_queue.get_nowait()
        except Exception as e:
            print(f"[ERROR] During TTS generation: {e}")

    def stop_audio_player(self):
        self.clear_queues()
        if self._generate_task and not self._generate_task.done():
            self._generate_task.cancel()

        # Clearing queues after cancelling generate_task to make sure
        # no audio was added to the audio queue while cancelling the task.
        self.clear_queues()
        self._tts_end_event.set()

        sd.get_stream().abort()

    def _chatterbox_build_student_conditionals(self) -> Conditionals | None:
        conds = None
        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or None,
            # Optional: Filter to download only what you need
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        )

        ckpt_dir = Path(local_path)

        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location="cpu").to(device)
        return conds

    def start(self):
        self._loop = asyncio.get_running_loop()
        self._worker_thread = threading.Thread(target=self._worker)
        self._worker_thread.daemon = True
        self._worker_thread.start()

    def _worker(self):
        self.audio_thread = threading.Thread(
            target=self.audio_player_worker, args=(self._audio_queue, self.model.sr)
        )
        self.audio_thread.daemon = True
        self.audio_thread.start()
        conds = self._chatterbox_build_student_conditionals()
        role = Role.TEACHER

        while True:
            try:
                if self._audio_queue.qsize() >= AUDIO_QUEUE_MAX_WAIT:
                    time.sleep(0.5)
                    continue

                try:
                    text_chunk = self.input_event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    time.sleep(0.5)
                    continue

                if text_chunk is None:
                    self._audio_queue.put(None)
                    self._tts_end_event.wait()
                    self._output_queue.put_nowait(TTSEndEvent.create(role))
                    self._tts_end_event.clear()
                    continue

                role = text_chunk.role
                audio_path = (
                    None if role == Role.TEACHER else "./voices/student_voice.wav"
                )

                if conds:
                    self.model.conds = conds

                initial_time = time.time()

                self._generate_task = asyncio.run_coroutine_threadsafe(
                    asyncio.to_thread(
                        self._generate_and_queue, text_chunk.text, audio_path
                    ),
                    self._loop,
                )
                try:
                    self._generate_task.result()
                except CancelledError:
                    print("[INFO] TTS was cancelled")
                print(f"[INFO] Chunk generation time = {time.time() - initial_time}")

            except Exception as e:
                print(f"ERROR: During TTS generation: {e}")
                traceback.print_exc()

    def clear_queues(self):
        while not self.input_event_queue.empty():
            try:
                self.input_event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break

    async def events(self):
        while True:
            event = await asyncio.to_thread(self._output_queue.get)
            yield event
