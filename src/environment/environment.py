import numpy as np
from gymnasium import spaces

from src.environment.core import BasicOpponent, HockeyEnvCore


class HockeyEnv(HockeyEnvCore):
    def __init__(self, weak: bool):
        super().__init__()
        self.opponent = BasicOpponent(weak=weak, keep_mode=self.keep_mode)
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

    def step(self, action):
        obs2 = self.obs_agent_two()
        action2 = self.opponent.act(obs2)
        actions_combined = np.hstack([action, action2])
        return super().step(actions_combined)
