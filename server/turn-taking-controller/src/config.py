from dataclasses import dataclass


@dataclass
class Config:
    USER_TURN_TIMEOUT: float = 1.0
    STUDENT_TURN_TIMEOUT: float = 1.0
    TEACHER_SERVER_HOST: str = "172.25.137.139"
    TEACHER_SERVER_PORT: int = 9000
    STUDENT_SERVER_HOST: str = "127.0.0.1"
    STUDENT_SERVER_PORT: int = 8001


config = Config()
