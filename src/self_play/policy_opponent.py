from uuid import uuid4

import numpy as np
from stable_baselines3.common.policies import BasePolicy

from src.environment import OpponentWrapper


class PolicyOpponent(OpponentWrapper):
    id: str = "policy"

    def __init__(self, policy: BasePolicy):
        self.id += f"_{uuid4()}"
        self.policy = policy

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.policy.predict(obs, deterministic=False)
        return action
