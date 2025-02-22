import time
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from typing import Iterator, Optional

import gymnasium as gym
import numpy as np
import torch
import wandb
from gymnasium import spaces
from torch.optim import Adam

from src.algos.common.replay_buffer import Batch, ReplayBuffer  # noqa: E402
from src.algos.sac.policy_base import ActorCriticBase
from src.utils.logx import EpochLogger  # noqa: E402


class Base(ABC):
    @property
    @abstractmethod
    def actor_critic_class(self) -> type[ActorCriticBase]:
        pass

    def __init__(
            self,
            env: gym.Env,
            replay_size: int,
            init_alpha: float,
            alpha_trainable: bool,
            actor_hidden_sizes: tuple[int, ...],
            critic_hidden_sizes: tuple[int, ...],
            device: str,
            policy_delay: int,
            batch_size: int,
            gamma: float,
            betas: tuple[float, float],
            lr: float,
            batch_norm_eps: Optional[float],
            batch_norm_momentum: Optional[float],
    ):
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                f"Expected Box observation space, got {type(env.observation_space)}"
            )

        if not isinstance(env.action_space, spaces.Box):
            raise TypeError(
                f"Expected Box action space, got {type(env.action_space)}"
            )

        self.env = env
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.gamma = gamma
        self.betas = betas
        self.lr = lr

        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Device: {self.device}")

        # Create actor-critic module
        self.ac = self.actor_critic_class(
            observation_space=env.observation_space,
            action_space=env.action_space,
            init_alpha=init_alpha,
            alpha_trainable=alpha_trainable,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )
        self.ac.to(self.device)

        # Init replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            max_size=replay_size,
            device=self.device,
        )

        # Init optimizers to None
        self._q_optimizer = None
        self._pi_optimizer = None
        self._alpha_optimizer = None

        # Init logger to None
        self._logger = None

    @property
    def q_params(self) -> Iterator[torch.nn.Parameter]:
        yield from chain(self.ac.q1.parameters(), self.ac.q2.parameters())

    @property
    def q_optimizer(self) -> Adam:
        if self._q_optimizer is None:
            raise ValueError("Optimizer not initialized")
        return self._q_optimizer

    @property
    def pi_optimizer(self) -> Adam:
        if self._pi_optimizer is None:
            raise ValueError("Optimizer not initialized")
        return self._pi_optimizer

    @property
    def alpha_optimizer(self) -> Adam:
        if self._alpha_optimizer is None:
            raise ValueError("Optimizer not initialized")
        return self._alpha_optimizer

    @property
    def logger(self) -> EpochLogger:
        if self._logger is None:
            raise ValueError("Logger not initialized")
        return self._logger

    def test_agent(self, test_env: gym.Env, num_test_episodes: int) -> None:
        returns, lengths, success = (
            np.zeros(num_test_episodes, dtype=np.float32),
            np.zeros(num_test_episodes, dtype=np.int32),
            np.zeros(num_test_episodes, dtype=np.bool),
        )
        for ep in range(num_test_episodes):
            obs, info = test_env.reset()
            ep_ret, ep_len = 0, 0
            terminated, truncated = False, False
            while not (terminated or truncated):
                action = self.ac.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                ep_ret += reward
                ep_len += 1

            returns[ep] = ep_ret
            lengths[ep] = ep_len
            success[ep] = info.get("is_success", False)

        self.logger.store(
            TestEpRet=float(returns.mean()),
            TestEpLen=float(lengths.mean()),
            TestSuccess=float(success.mean()),
        )

    def save_model(self, model_path: Path, save_buffer: bool = False) -> None:
        if model_path.is_dir():
            raise ValueError(f"Path {model_path} is a directory")

        model_path = model_path.with_suffix(".pth")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.ac.state_dict(), model_path)

        if save_buffer:
            buffer_path = model_path.with_suffix(".buffer.pth")
            self.replay_buffer.save(buffer_path)

    @classmethod
    def load_model(
            cls,
            env: gym.Env,
            model_path: Path,
            buffer_path: Optional[Path] = None,
            **kwargs,
    ) -> "Base":
        model_obj = cls(env, **kwargs)
        state_dict = torch.load(model_path)
        model_obj.ac.load_state_dict(state_dict)
        if buffer_path is not None:
            model_obj.replay_buffer.load(buffer_path)
        return model_obj

    @abstractmethod
    def compute_loss_q(self, batch: Batch) -> torch.Tensor:
        """
        Compute the loss for the Q-networks.
        """
        pass

    def compute_loss_pi(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        pi, log_p_pi = self.ac.pi(batch.obs)
        q1_pi = self.ac.q1(batch.obs, pi)
        q2_pi = self.ac.q2(batch.obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (self.ac.log_alpha.exp() * log_p_pi - q_pi).mean()

        return loss_pi, log_p_pi

    def update(self, batch: Batch, update_policy: bool) -> None:
        loss_q = self.update_critics(batch)
        logger_update_dict = dict(LossQ=loss_q)

        if update_policy:
            loss_pi, loss_alpha = self.update_policy(batch)
            logger_update_dict["LossPi"] = loss_pi
            logger_update_dict["LossAlpha"] = loss_alpha

        logger_update_dict["Alpha"] = self.ac.log_alpha.exp().item()
        self.logger.store(**logger_update_dict)

    def update_critics(self, batch: Batch) -> float:
        for p in self.q_params:
            p.requires_grad = True

        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(batch)
        loss_q.backward()
        self.q_optimizer.step()

        return loss_q.item()

    def update_policy(self, batch: Batch) -> tuple[float, float]:
        for p in self.q_params:
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi, log_p_pi = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Update alpha if it is trainable
        if self.ac.log_alpha.requires_grad:
            self.alpha_optimizer.zero_grad()
            loss_alpha = -(
                    self.ac.log_alpha * (log_p_pi.detach() - float(self.act_dim))
            ).mean()
            loss_alpha.backward()
            self.alpha_optimizer.step()

            return loss_pi.item(), loss_alpha.item()
        else:
            return loss_pi.item(), 0.0

    def learn(
            self,
            total_steps: int,
            warmup_steps: Optional[int] = 10_000,
            test_env: Optional[gym.Env] = None,
            num_test_episodes: Optional[int] = 10,
            logging_steps: Optional[int] = 10_000,
            save_freq: Optional[int] = 10,
            seed: Optional[int] = None,
            wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    ) -> None:
        self._logger = EpochLogger(wandb_run=wandb_run)

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

            if warmup_steps is None or t >= warmup_steps:
                batch = self.replay_buffer.sample_batch(self.batch_size)
                update_policy = (t + 1) % self.policy_delay == 0
                self.update(batch, update_policy)

            if save_freq and (t + 1) % save_freq == 0:
                self.save_model(Path(f"models/cross_q_{t}"))

            if logging_steps and (t + 1) % logging_steps == 0:
                if num_test_episodes and test_env is not None:
                    self.test_agent(test_env, num_test_episodes)

                self.logger.log_tabular("EpRet", with_min_and_max=True)
                self.logger.log_tabular("TestEpRet", with_min_and_max=True)
                self.logger.log_tabular("EpLen", average_only=True)
                self.logger.log_tabular("TestEpLen", average_only=True)
                self.logger.log_tabular("LossPi", average_only=True)
                self.logger.log_tabular("LossQ", average_only=True)
                self.logger.log_tabular("FPS", t / (time.time() - start_time))
                self.logger.log_tabular("Time", time.time() - start_time)
                self.logger.dump_tabular()
