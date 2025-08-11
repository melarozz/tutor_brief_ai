from dataclasses import dataclass
import numpy as np

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = None
    embedding: np.ndarray = None
