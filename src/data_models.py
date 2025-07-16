from dataclasses import dataclass, asdict

@dataclass
class Utterance:
    """Individual utterance with speaker and timing"""
    episode_id: str
    start: float
    end: float
    speaker: str
    text: str