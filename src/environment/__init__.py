from typing import Optional

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from src.environment.hockey_env import OpponentWrapper


def make_hockey_env(opponent: Optional[OpponentWrapper] = None) -> gym.Env:
    if "Hockey-v0" not in gym.envs.registry:
        gym.register(
            id="Hockey-v0",
            entry_point="src.environment.hockey_env:HockeyEnv",
        )
    return gym.make("Hockey-v0", opponent=opponent)


def make_vec_hockey_env(
    n_envs: int, opponent: Optional[OpponentWrapper] = None
) -> VecEnv:
    return make_vec_env(
        lambda: make_hockey_env(opponent),
        n_envs=n_envs,
        # vec_env_cls=SubprocVecEnv,
    )
