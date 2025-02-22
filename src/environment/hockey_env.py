from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.environment.core import BasicOpponent as _BasicOpponent, HockeyEnv2Player


class OpponentWrapper(ABC):
    id: str = ""

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        ...


class BasicOpponent(_BasicOpponent, OpponentWrapper):
    id: str = "basic"

    def __init__(self, weak: bool = True):
        super().__init__(weak=weak)


class HockeyEnv(HockeyEnv2Player):
    """
    Hockey environment for single agent and fixed opponent.
    Initialization function
    """

    def __init__(self, opponent: Optional[OpponentWrapper]):
        super().__init__()
        self._opponent = opponent or BasicOpponent()

    def step(self, action) -> tuple:
        obs2 = self.obs_agent_two()
        action2 = self._opponent.act(obs2)
        actions_combined = np.hstack([action, action2])
        return super().step(actions_combined)

    def set_opponent(self, opponent: OpponentWrapper) -> None:
        self._opponent = opponent

    def opponent_id(self) -> str:
        return str(self._opponent.id)
