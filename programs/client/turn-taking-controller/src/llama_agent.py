from collections.abc import AsyncIterator

import shared_lib.events


class LlamaAgent:
    def __init__(self):
        print("Llama agent")

    async def process_turn(
        self, event: AsyncIterator[shared_lib.events.STTEvent]
    ) -> AsyncIterator[shared_lib.events.TurnTakingEvent]:
        yield shared_lib.events.STTChunkEvent.create("TEST")
