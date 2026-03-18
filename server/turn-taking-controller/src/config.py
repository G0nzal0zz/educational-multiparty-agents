from dataclasses import dataclass


@dataclass
class Config:
    HOST: str = "127.0.0.1"
    PORT: int = 9000
    USER_TURN_TIMEOUT: float = 1.0
    STUDENT_TURN_TIMEOUT: float = 1.0


config = Config()
