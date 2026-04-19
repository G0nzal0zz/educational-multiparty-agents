from client_lib.events import AgentChunkEvent, AgentEndEvent
from client_lib.ollama_llm import PHRASES_IN_CHUNK

INIIAL_LESSON = """
Simpson’s Paradox is a statistical phenomenon where a trend that appears within several separate groups reverses or disappears when those groups are combined into a single dataset. In other words, looking only at the overall data can lead to a completely different conclusion than looking at the data broken down into meaningful subgroups.

A well-known example comes from university admissions data. When all applications were combined, it appeared that one gender was being treated unfairly. However, when the data was separated by department, the pattern either disappeared or reversed. This happened because different departments had different acceptance rates, and applicants were unevenly distributed across those departments.

This leads to the key idea of confounding variables. A confounding variable is an external factor that influences the relationship between the variables being studied. In Simpson’s Paradox, these hidden variables explain why the trend changes when data is aggregated. They do not directly “cause” the paradox, but they create the conditions where the reversal appears.

A common misconception is that Simpson’s Paradox is a mistake or a statistical error that can be fixed by simply collecting more data or applying a formula. In reality, it is not an error. It is a real effect that arises from how data is grouped and analyzed, and it requires careful interpretation rather than correction.

Recognizing Simpson’s Paradox is especially important in high-stakes situations such as analyzing clinical trial data. If researchers only look at overall results, they may draw incorrect conclusions about whether a treatment is effective. By examining subgroup data, they can avoid misleading interpretations.

The first step in identifying Simpson’s Paradox is to examine trends within individual groups before looking at the aggregated data. This helps reveal whether the overall pattern is hiding important differences between subgroups.

Finally, Simpson’s Paradox highlights why correlation does not necessarily imply causation. Even if a strong relationship appears in the data, it may be misleading due to hidden variables. The reversal of trends between subgroup and aggregate data shows that observed correlations alone are not enough to establish a true causal relationship.
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
