import asyncio
import os
import queue
import threading
import time
import traceback
from pathlib import Path

import sounddevice as sd
import torch
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
            # sd.wait()  # Wait for this chunk to finish before returning

        except Exception as e:
            print(f"ERROR: Playing audio: {e}")

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
                print(f"ERROR: Audio player: {e}")

    def _generate_and_queue(self, text: str, audio_path: str | None):
        """Sync wrapper that runs model.generate and puts chunks in queue"""
        print("INFO: Starting audio generation")
        try:
            for audio_chunk in self.model.generate(
                text=text,
                audio_prompt_path=audio_path,
                temperature=0.8,
                cfg_weight=0.5,
            ):
                self._audio_queue.put(audio_chunk.clone())
        except Exception as e:
            if not self.input_event_queue.empty():
                self.input_event_queue.get_nowait()
            print(f"ERROR: During TTS generation: {e}")
            traceback.print_exc()
        finally:
            print("INFO: Generation finished or cancelled")

    def stop_audio_player(self):
        print("INFO: Stop audio player called")
        self.clear_queues()
        if self._generate_task and not self._generate_task.done():
            print("INFO: Cancelling generation task")
            self._generate_task.cancel()
        else:
            print("INFO: No active generation task to cancel")

        print("self.audio_thread.is_alive(): ", self.audio_thread.is_alive())

        if self.audio_thread is None or self.audio_thread.is_alive() is False:
            print("INFO: Audio thread not running")

        sd.get_stream().abort()
        print("INFO: Stopping audio player")

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
                    print(
                        "INFO: Waiting audio to be played before processing more text chunk"
                    )
                    time.sleep(1)
                    continue

                try:
                    text_chunk = self.input_event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    time.sleep(0.1)
                    continue

                if text_chunk is None:
                    self._audio_queue.put(None)
                    self._tts_end_event.wait()
                    self._loop.call_soon_threadsafe(
                        self._output_queue.put_nowait, TTSEndEvent.create(role)
                    )
                    self._tts_end_event.clear()
                    print("INFO: Audio finished playing")
                    continue

                role = text_chunk.role
                audio_path = None if role == Role.TEACHER else "./student_voice.wav"

                if conds:
                    self.model.conds = conds

                initial_time = time.time()
                print(
                    f"INFO: Starting generation for text chunk: {text_chunk.text[:50]}..."
                )
                self._generate_task = asyncio.run_coroutine_threadsafe(
                    asyncio.to_thread(
                        self._generate_and_queue, text_chunk.text, audio_path
                    ),
                    self._loop,
                )
                try:
                    self._generate_task.result()
                except asyncio.CancelledError:
                    print("INFO: Generation was cancelled")
                print(f"INFO: Chunk generation time = {time.time() - initial_time}")

            except Exception as e:
                print(f"ERROR: During TTS generation: {e}")
                traceback.print_exc()

    def clear_queues(self):
        while not self.input_event_queue.empty():
            try:
                self.input_event_queue.get_nowait()
                print("[TTS] Emptying input event queue ")
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
