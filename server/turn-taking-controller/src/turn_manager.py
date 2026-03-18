from enum import Enum

from shared_lib.events import Role


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

    def is_role_turn(self, role: Role) -> bool:
        return self._current_turn.value == role.value
