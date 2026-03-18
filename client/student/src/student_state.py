from dataclasses import dataclass, field


@dataclass
class StudentState:
    transcript: list[dict[str, str]] = field(default_factory=list)
    questions_asked: list[str] = field(default_factory=list)
    human_level_hints: list[str] = field(default_factory=list)

    def append_transcript(self, role: str, text: str) -> None:
        if self.transcript and self.transcript[-1]["role"] == role:
            self.transcript[-1]["text"] += " " + text.strip()
        else:
            self.transcript.append({"role": role, "text": text.strip()})

    def note_human_input(self, text: str) -> None:
        hint = _infer_level_hint(text)
        if hint:
            self.human_level_hints.append(hint)


def _infer_level_hint(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["what is", "what does", "what are", "i don't understand", "i dont understand"]):
        return "The human student asked a basic definitional question, suggesting a beginner level."
    if any(w in text_lower for w in ["why", "how come", "how does", "explain"]):
        return "The human student asked a conceptual why or how question, suggesting intermediate curiosity."
    if any(w in text_lower for w in ["compare", "difference", "versus", "trade-off", "tradeoff", "implication"]):
        return "The human student asked an analytical or comparative question, suggesting a higher level of engagement."
    if len(text.split()) <= 4:
        return "The human student gave a very short response, possibly indicating confusion or disengagement."
    return ""
