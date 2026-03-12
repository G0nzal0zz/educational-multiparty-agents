from dataclasses import dataclass


@dataclass
class Config:
    TTC_SERVER_HOST: str = "127.0.0.1"
    TTC_SERVER_PORT: int = 9000


config = Config()
