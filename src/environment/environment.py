from typing import Optional

import numpy as np
import torch
from gymnasium import spaces

from src.algos.core.actor import SquashedGaussianMLPActor
from src.environment.core import BasicOpponent, HockeyEnvCore


class HockeyEnv(HockeyEnvCore):
    def __init__(self, weak: bool):
        super().__init__()
        self.use_opponent = True
        self.opponent = BasicOpponent(weak=weak, keep_mode=self.keep_mode)
        self.actor: Optional[SquashedGaussianMLPActor] = None
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

    def step(self, action):
        obs2 = self.obs_agent_two()
        if self.use_opponent:
            action2 = self.opponent.act(obs2)
        elif self.actor is not None:
            obs2 = torch.as_tensor(obs2, dtype=torch.float32, device=self.actor.device)
            with torch.no_grad():
                action2, _ = (
                    self.actor(obs2, deterministic=False, with_logprob=False)
                    .cpu()
                    .numpy()
                )
        else:
            raise ValueError("Actor must be provided if not using opponent")

        actions_combined = np.hstack([action, action2])
        return super().step(actions_combined)

    def set(
        self,
        use_opponent: bool,
        weak: Optional[bool],
        actor: Optional[SquashedGaussianMLPActor],
    ):
        self.use_opponent = use_opponent

        if weak and weak != self.opponent.weak:
            self.opponent = BasicOpponent(weak=weak, keep_mode=self.keep_mode)

        if actor is not None:
            self.actor = actor
