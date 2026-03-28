from dataclasses import dataclass


@dataclass
class Config:
    HOST: str = "0.0.0.0"
    PORT: int = 9000
    USER_TURN_TIMEOUT: float = 8.0


config = Config()
