import sys
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import wandb
from gymnasium import spaces
from torch.optim import Adam

sys.path.append(str(Path(__file__).parents[2].resolve()))

import src.cross_q.cross_q_policy as cross_q_policy  # noqa: E402
from src.cross_q.cross_q_replay_buffer import Batch, ReplayBuffer  # noqa: E402
from src.logx import EpochLogger  # noqa: E402


class CrossQ:
    """
    Cross Q-learning with a BN-equipped critic that uses a joint forward pass.
    This removes the need for target networks by concatenating (s, a) and (s', a') batches,
    ensuring that the BatchNorm layers see a mixture of both distributions.
    """

    def __init__(
        self,
        env: gym.Env,
        ac_kwargs: dict,
        replay_size: int,
        device: str = None,
    ):
        self.env = env

        if not isinstance(self.env.observation_space, spaces.Box):
            raise TypeError(
                f"Expected Box observation space, got {type(self.env.observation_space)}"
            )

        if not isinstance(self.env.action_space, spaces.Box):
            raise TypeError(
                f"Expected Box action space, got {type(self.env.action_space)}"
            )

        observation_space: spaces.Box = self.env.observation_space
        action_space: spaces.Box = self.env.action_space

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.target_entropy = -float(self.act_dim)

        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Device: {self.device}")

        # Create actor-critic module
        self.ac = cross_q_policy.MLPActorCritic(
            observation_space, action_space, **ac_kwargs
        ).to(self.device)

        # List of parameters for both Q-networks
        self.q_params = tuple(self.ac.q1.parameters()) + tuple(self.ac.q2.parameters())

        # Experience buffer
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

        # Logger
        self._logger = None

        # Count variables
        var_counts = tuple(
            sum(p.numel() for p in module.parameters())
            for module in [self.ac.pi, self.ac.q1, self.ac.q2]
        )
        print(
            f"\nNumber of parameters: \t pi: {var_counts[0]}, \t q1: {var_counts[1]}, \t q2: {var_counts[2]}\n"
        )

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

    def compute_loss_q(self, batch: Batch, gamma: float) -> torch.Tensor:
        with torch.no_grad():
            # Get next actions and corresponding log-probs
            a2, log_p_a2 = self.ac.pi(batch.obs2)

        # Use the joint forward pass so that BatchNorm sees both current and next samples
        q1_current, q1_next = self.ac.q1.forward_joint(
            batch.obs, batch.act, batch.obs2, a2
        )
        q2_current, q2_next = self.ac.q2.forward_joint(
            batch.obs, batch.act, batch.obs2, a2
        )

        # Compute the target backup without gradients using adaptive alpha
        with torch.no_grad():
            q_pi = torch.min(q1_next, q2_next)
            backup = batch.reward + gamma * (1 - batch.done) * (
                q_pi - self.ac.log_alpha.exp() * log_p_a2
            )

        loss_q1 = ((q1_current - backup) ** 2).mean()
        loss_q2 = ((q2_current - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        pi, log_p_pi = self.ac.pi(batch.obs)
        q1_pi = self.ac.q1(batch.obs, pi)
        q2_pi = self.ac.q2(batch.obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (self.ac.log_alpha.exp() * log_p_pi - q_pi).mean()

        return loss_pi, log_p_pi

    def update(self, batch: Batch, gamma: float) -> None:
        # First update the Q-networks using the joint forward pass
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(batch, gamma)
        loss_q.backward()
        self.q_optimizer.step()
        self.logger.store(LossQ=loss_q.item())

        # Freeze Q-networks while updating the policy
        for p in self.q_params:
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi, log_p_pi = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.q_params:
            p.requires_grad = True

        self.logger.store(LossPi=loss_pi.item())

        # Adaptive alpha update
        self.alpha_optimizer.zero_grad()
        loss_alpha = -(
            self.ac.log_alpha * (log_p_pi.detach() + self.target_entropy)
        ).mean()
        loss_alpha.backward()
        self.alpha_optimizer.step()

        self.logger.store(
            Alpha=self.ac.log_alpha.exp().item(),
            LossAlpha=loss_alpha.item(),
        )

    def test_agent(self, test_env: gym.Env, num_test_episodes: int) -> None:
        returns, lengths, success = (
            np.zeros(num_test_episodes),
            np.zeros(num_test_episodes),
            np.zeros(num_test_episodes),
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
            TestEpRet=returns.mean(),
            TestEpLen=lengths.mean(),
            TestSuccess=success.mean(),
        )

    def save_model(self, path: Path, save_buffer: bool = False) -> None:
        if path.is_dir():
            raise ValueError(f"Path {path} is a directory")

        model_path = path.with_suffix(".pth")
        torch.save(self.ac.state_dict(), model_path)

        if save_buffer:
            buffer_path = path.parent / f"{path.stem}_buffer"
            self.replay_buffer.save(buffer_path)

    @classmethod
    def load_model(
        cls, env: gym.Env, path: Path, buffer_path: Optional[Path] = None, **kwargs
    ) -> "CrossQ":
        model_ = cls(env, **kwargs)
        model_.ac.load_state_dict(torch.load(path))
        if buffer_path is not None:
            model_.replay_buffer.load(buffer_path)
        return model_

    def learn(
        self,
        epochs: int,
        steps_per_epoch: int,
        start_steps: int,
        update_after: int,
        update_every: int,
        batch_size: int,
        gamma: float,
        seed: int,
        betas: tuple[float, float],
        lr: float,
        test_env: gym.Env,
        num_test_episodes: int,
        save_freq: int,
        wandb_run: Optional[wandb.sdk.wandb_run.Run],
    ) -> None:
        self._logger = EpochLogger(wandb_run=wandb_run)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self._q_optimizer = Adam(self.q_params, lr=lr, betas=betas)
        self._pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr, betas=betas)
        self._alpha_optimizer = Adam([self.ac.log_alpha], lr=lr, betas=betas)

        # Main interaction loop
        total_steps = steps_per_epoch * epochs
        start_time = time.time()
        obs, _ = self.env.reset()
        ep_ret, ep_len = 0, 0

        for t in range(total_steps):
            self.logger.set_step(t)

            if t >= start_steps:
                action = self.ac.act(obs, deterministic=False)
            else:
                action = self.env.action_space.sample()

            obs2, reward, terminated, truncated, info = self.env.step(action)
            ep_ret += reward
            ep_len += 1

            # Store if episode ended. However, if terminated only because of truncation,
            # we don't store the done transition because it may continue.
            self.replay_buffer.store(
                obs=obs,
                act=action,
                rew=reward,
                obs2=obs2,
                done=terminated and not truncated,
            )
            obs = obs2

            if terminated or truncated:
                self.logger.store(
                    EpRet=ep_ret,
                    EpLen=ep_len,
                    EpSuccess=int(info.get("is_success", False)),
                )
                obs, info = self.env.reset()
                ep_ret, ep_len = 0, 0

            if t >= update_after and t % update_every == 0:
                for _ in range(update_every):
                    batch = self.replay_buffer.sample_batch(batch_size)
                    self.update(batch, gamma)

            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch

                if (epoch % save_freq == 0) or (epoch == epochs):
                    self.save_model(Path(f"models/cross_q_{epoch}"))

                self.test_agent(test_env, num_test_episodes)

                self.logger.log_tabular("Epoch", epoch)
                self.logger.log_tabular("EpRet", with_min_and_max=True)
                self.logger.log_tabular("TestEpRet", with_min_and_max=True)
                self.logger.log_tabular("EpLen", average_only=True)
                self.logger.log_tabular("TestEpLen", average_only=True)
                self.logger.log_tabular("TotalEnvInteracts", t)
                self.logger.log_tabular("LossPi", average_only=True)
                self.logger.log_tabular("LossQ", average_only=True)
                self.logger.log_tabular("FPS", t / (time.time() - start_time))
                self.logger.log_tabular("Time", time.time() - start_time)

                self.logger.dump_tabular()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="Hockey-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps_per_epoch", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--replay_size", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_steps", type=int, default=1000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--num_test_episodes", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.env == "Hockey-v0":
        gym.register(
            id="Hockey-v0",
            entry_point="src.environment.environment:HockeyEnv",
        )

    model = CrossQ(
        env=gym.make(args.env),
        ac_kwargs=dict(),
        replay_size=args.replay_size,
        device=args.device,
    )

    run, error = wandb.init(project="cross_q", config=args.__dict__), None
    try:
        model.learn(
            seed=args.seed,
            test_env=gym.make(args.env),
            num_test_episodes=args.num_test_episodes,
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
            gamma=args.gamma,
            lr=args.lr,
            betas=(0.5, 0.999),
            batch_size=args.batch_size,
            start_steps=args.start_steps,
            update_after=args.update_after,
            update_every=args.update_every,
            save_freq=args.save_freq,
            wandb_run=run,
        )
    except (KeyboardInterrupt, Exception) as e:
        print(e)
        error = e
    finally:
        run.finish(exit_code=0 if error is None else 1)
