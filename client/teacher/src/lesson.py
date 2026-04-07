from client_lib.events import AgentChunkEvent, AgentEndEvent
from client_lib.ollama_llm import PHRASES_IN_CHUNK

INIIAL_LESSON = """
Simpson’s Paradox is a statistical phenomenon where a trend that appears within several separate groups reverses or disappears when those groups are combined into a single dataset. In other words, looking only at the overall data can lead to a completely different conclusion than looking at the data broken down into meaningful subgroups.
"""


def lesson_generator():
    phrases = ""
    phrases_in_chunk = 0

    for c in INIIAL_LESSON:
        phrases = phrases + c
        if c in ".!?":
            phrases_in_chunk = phrases_in_chunk + 1
        if phrases_in_chunk >= PHRASES_IN_CHUNK:
            phrases_in_chunk = 0
            yield AgentChunkEvent.create(phrases)
            phrases = ""
    if len(phrases):
        yield AgentChunkEvent.create(phrases)
    yield AgentEndEvent.create(INIIAL_LESSON)
