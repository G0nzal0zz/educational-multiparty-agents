import asyncio

from chatterbox.tts import ChatterboxTTS as Chatterbox
from shared_lib.events import Role, SocketAgentAudioChunkEvent

model = Chatterbox.from_pretrained(device="cuda")


class ChatterboxTTS:
    async def generate(self, text: str):
        print("text: ", text)

        # TODO: Enable streaming
        result = await asyncio.to_thread(model.generate, text)

        audio = result.detach().cpu().numpy()
        audio = audio.squeeze()

        return SocketAgentAudioChunkEvent.create(
            role=Role.TEACHER, audio=audio.tobytes()
        )
