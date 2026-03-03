import asyncio

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS as Chatterbox

from server_lib.events import TTSChunkEvent

model = Chatterbox.from_pretrained(device="cuda")


class ChatterboxTTS:
    async def generate(self, text: str):
        # Run TTS in thread to avoid blocking event loop
        print("text: ", text)
        result = await asyncio.to_thread(model.generate, text)  # PyTorch tensor

        ta.save("test-audio.wav", result, model.sr)

        audio = result.detach().cpu().numpy()  # float32
        audio = audio.squeeze()  # -> (54720,)
        print(audio.shape)

        # Play audio with correct samplerate
        # sd.play(audio, samplerate=model.sr, blocking=True)  # usually 22050 or 24000

        # If you want to stream over network, serialize as bytes
        return TTSChunkEvent.create(audio.tobytes())
