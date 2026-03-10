from enum import Enum


class Turn(Enum):
    HUMAN = 1
    TEACHER = 2
    STUDENT = 3
    IDLE = 4


class TurnManager:
    def __init__(self):
        self._current_turn: Turn = Turn.IDLE

    @property
    def current_turn(self) -> Turn:
        return self._current_turn

    def set_turn(self, turn: Turn) -> None:
        self._current_turn = turn
