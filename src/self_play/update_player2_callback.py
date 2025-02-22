import random
from collections import deque
from copy import deepcopy

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.environment.hockey_env import BasicOpponent, OpponentWrapper
from src.self_play.policy_opponent import PolicyOpponent

NUM_POLICIES = 10
WEAK_OPPONENT_PROP = 0.05
STRONG_OPPONENT_PROP = 0.5
WEAK_OPPONENT = BasicOpponent(weak=True)
STRONG_OPPONENT = BasicOpponent(weak=False)


class SelfPlayCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.opponents: deque[PolicyOpponent] = deque(maxlen=NUM_POLICIES)
        self.last_checkpoint_time = self.num_timesteps

    @property
    def checkpoint_freq(self):
        if self.num_timesteps < 2_000_000:
            return 200_000
        if self.num_timesteps < 4_000_000:
            return 400_000
        if self.num_timesteps < 8_000_000:
            return 800_000
        if self.num_timesteps < 10_000_000:
            return 1_000_000
        return 1_000_000_000  # Never checkpoint

    def sample_opponent(self) -> OpponentWrapper:
        if random.random() < WEAK_OPPONENT_PROP:
            return WEAK_OPPONENT
        elif random.random() < STRONG_OPPONENT_PROP:
            return STRONG_OPPONENT
        elif len(self.opponents) < self.opponents.maxlen // 2:
            return STRONG_OPPONENT
        return random.choice(self.opponents)

    def _on_rollout_start(self) -> None:
        if (self.num_timesteps - self.last_checkpoint_time) >= self.checkpoint_freq:
            policy = deepcopy(self.model.policy)
            policy.to(self.model.device)
            policy_opponent = PolicyOpponent(policy=policy, timestep=self.num_timesteps)
            self.opponents.append(policy_opponent)
            self.last_checkpoint_time = self.num_timesteps
            print(
                f"Checkpointing policy at {self.num_timesteps} timesteps. "
                f"Number of policies: {len(self.opponents)}. "
                f"Checkpoint freq: {self.checkpoint_freq}."
            )
            print(self.training_env.env_method("opponent_id"))

        opponent = self.sample_opponent()
        index = random.randint(0, self.training_env.num_envs - 1)
        self.training_env.env_method(
            "set_opponent",
            opponent,
            indices=index,
        )

    def _on_step(self) -> bool:
        return True
