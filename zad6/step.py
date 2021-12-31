from dataclasses import dataclass


@dataclass
class Step:
    reward: float
    old_state_index: int
    next_state_index: int
    finished: bool
