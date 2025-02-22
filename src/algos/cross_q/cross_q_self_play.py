import copy
import random
import time
from collections import deque
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import wandb
from torch.optim import Adam

from src.algos.core import SquashedGaussianMLPActor
from src.utils.logx import EpochLogger
from .cross_q import CrossQ


class CrossQSelfPlay(CrossQ):
    def __init__(
        self,
        env: gym.Env,
        basic_opponent_prob: float = 0.5,
        weak_opponent_prob: float = 0.1,
        checkpoint_pool_size: int = 10,
        policy_checkpoint_freq: int = 100_000,
        **kwargs,
    ):
        super().__init__(env=env, **kwargs)
        self.basic_opponent_prob = basic_opponent_prob
        self.weak_opponent_prob = weak_opponent_prob
        self.policy_checkpoint_freq = policy_checkpoint_freq
        self.checkpoint_pool = deque(maxlen=checkpoint_pool_size)

    def add_to_pool(self, actor: SquashedGaussianMLPActor) -> None:
        checkpoint = copy.deepcopy(actor)
        checkpoint.to(self.device)
        checkpoint.eval()
        self.checkpoint_pool.append(checkpoint)

    def sample_from_pool(self) -> Optional[SquashedGaussianMLPActor]:
        if len(self.checkpoint_pool) < self.checkpoint_pool.maxlen // 2:
            return None
        if np.random.rand() < self.basic_opponent_prob:
            return None
        return random.choice(self.checkpoint_pool)

    def learn(
        self,
        total_steps: int,
        warmup_steps: Optional[int] = 1_000,
        test_env: Optional[gym.Env] = None,
        num_test_episodes: Optional[int] = 10,
        logging_steps: Optional[int] = 10_000,
        save_freq: Optional[int] = 100_000,
        seed: Optional[int] = None,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    ) -> None:
        if not hasattr(self.env.unwrapped, "set"):
            raise ValueError("Environment must be HockeyEnv")

        self._logger = EpochLogger(wandb_run=wandb_run)
        self._logger.update_config(
            **self.config,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            test_env=test_env.unwrapped.spec.id if test_env is not None else None,
            num_test_episodes=num_test_episodes,
            logging_steps=logging_steps,
            save_freq=save_freq,
            seed=seed,
        )

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self._q_optimizer = Adam(self.q_params, lr=self.lr, betas=self.betas)
        self._pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr, betas=self.betas)
        self._alpha_optimizer = Adam([self.ac.log_alpha], lr=self.lr, betas=self.betas)

        # Main interaction loop
        start_time = time.time()
        obs, _ = self.env.reset()
        ep_ret, ep_len = 0, 0

        for t in range(total_steps):
            self.logger.set_step(t)

            if warmup_steps is None or t >= warmup_steps:
                action = self.ac.act(obs, deterministic=False)
            else:
                action = self.env.action_space.sample()

            obs2, reward, terminated, truncated, info = self.env.step(action)
            ep_ret += reward
            ep_len += 1

            self.replay_buffer.store(
                obs=obs,
                act=action,
                rew=reward,
                obs2=obs2,
                done=terminated and not truncated,
            )
            obs = obs2

            if terminated or truncated:
                ep_success = int(info.get("is_success", False))
                self.logger.store(EpRet=ep_ret, EpLen=ep_len, EpSuccess=ep_success)
                obs, info = self.env.reset()
                ep_ret, ep_len = 0, 0

                if actor := self.sample_from_pool():
                    self.env.unwrapped.set(use_opponent=False, weak=None, actor=actor)
                else:
                    weak = np.random.rand() < self.weak_opponent_prob
                    self.env.unwrapped.set(use_opponent=True, weak=weak, actor=None)

            if (t + 1) % self.policy_checkpoint_freq == 0:
                self.add_to_pool(actor=self.ac.pi)

            if warmup_steps is None or t >= warmup_steps:
                batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
                update_policy = (t + 1) % self.policy_delay == 0
                self.update(batch=batch, update_policy=update_policy)

            if save_freq and (t + 1) % save_freq == 0:
                run_id = wandb_run.id if wandb_run is not None else "local"
                model_path = Path(f"models/{run_id}/model_{t}.pth")
                self.save_model(model_path=model_path, save_buffer=False)

            if logging_steps and (t + 1) % logging_steps == 0:
                if num_test_episodes and test_env is not None:
                    self.test_agent(
                        test_env=test_env,
                        num_test_episodes=num_test_episodes,
                    )

                self.logger.log_tabular("EpRet", with_min_and_max=True)
                self.logger.log_tabular("TestEpRet", with_min_and_max=True)
                self.logger.log_tabular("EpLen", average_only=True)
                self.logger.log_tabular("TestEpLen", average_only=True)
                self.logger.log_tabular("LossPi", average_only=True)
                self.logger.log_tabular("LossQ", average_only=True)
                self.logger.log_tabular("FPS", t / (time.time() - start_time))
                self.logger.log_tabular("Time", time.time() - start_time)
                self.logger.dump_tabular()
