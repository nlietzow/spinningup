import itertools
import sys
import time
from pathlib import Path
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch
from torch.optim import Adam

sys.path.append(str(Path(__file__).parents[2].resolve()))

import src.cross_q.core as core
from src.config import setup_logger_kwargs
from src.utils.logx import EpochLogger


class Batch(NamedTuple):
    obs: torch.Tensor
    act: torch.Tensor
    reward: torch.Tensor
    obs2: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents with GPU storage.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        # Initialize tensors directly on the specified device
        self.obs_buf = torch.zeros(
            core.combined_shape(size, obs_dim), dtype=torch.float32
        )
        self.obs2_buf = torch.zeros(
            core.combined_shape(size, obs_dim), dtype=torch.float32
        )
        self.act_buf = torch.zeros(
            core.combined_shape(size, act_dim), dtype=torch.float32
        )
        self.rew_buf = torch.zeros(size, dtype=torch.float32)
        self.done_buf = torch.zeros(size, dtype=torch.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def to_tensor(self, array):
        return torch.tensor(array, device=self.device)

    def store(self, obs, act, rew, obs2, done):
        self.obs_buf[self.ptr] = self.to_tensor(obs)
        self.obs2_buf[self.ptr] = self.to_tensor(obs2)
        self.act_buf[self.ptr] = self.to_tensor(act)
        self.rew_buf[self.ptr] = self.to_tensor(rew)
        self.done_buf[self.ptr] = self.to_tensor(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size) -> Batch:
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return Batch(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            reward=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )


def cross_q(
    env_fn,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=5_000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    lr=1e-3,
    alpha=0.1,
    batch_size=256,
    start_steps=1000,
    update_after=1000,
    update_every=10,
    num_test_episodes=10,
    logger_kwargs=dict(),
    save_freq=10,
    device=None,
):
    """
    Cross Q-learning with a BN-equipped critic that uses a joint forward pass.
    This removes the need for target networks by concatenating (s, a) and (s', a') batches,
    ensuring that the BatchNorm layers see a mixture of both distributions.
    """
    local_vars = locals()
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(local_vars)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    if device is None or device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.log("Device: %s" % device)

    # Create actor-critic module
    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    ac.to(device)

    # List of parameters for both Q-networks
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device
    )

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log("\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts)

    # Set up function for computing CrossQ Q-losses with joint forward pass
    def compute_loss_q(batch: Batch):
        with torch.no_grad():
            # Get next actions and corresponding log-probs
            a2, logp_a2 = ac.pi(batch.obs2)

        # Use the joint forward pass so that BatchNorm sees both current and next samples
        q1_current, q1_next = ac.q1.forward_joint(batch.obs, batch.act, batch.obs2, a2)
        q2_current, q2_next = ac.q2.forward_joint(batch.obs, batch.act, batch.obs2, a2)

        # Compute the target backup without gradients
        with torch.no_grad():
            q_pi = torch.min(q1_next, q2_next)
            backup = batch.reward + gamma * (1 - batch.done) * (q_pi - alpha * logp_a2)

        loss_q1 = ((q1_current - backup) ** 2).mean()
        loss_q2 = ((q2_current - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(
            Q1Vals=q1_current.detach().cpu().numpy(),
            Q2Vals=q2_current.detach().cpu().numpy(),
        )

        return loss_q, q_info

    # Set up function for computing CrossQ pi loss
    def compute_loss_pi(batch: Batch):
        pi, logp_pi = ac.pi(batch.obs)
        q1_pi = ac.q1(batch.obs, pi)
        q2_pi = ac.q2(batch.obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (alpha * logp_pi - q_pi).mean()

        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())
        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr, betas=(0.5, 0.999))
    q_optimizer = Adam(q_params, lr=lr, betas=(0.5, 0.999))

    def update(batch: Batch):
        # First update the Q-networks using the joint forward pass
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(batch)
        loss_q.backward()
        q_optimizer.step()
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks while updating the policy
        for p in q_params:
            p.requires_grad = False

        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(batch)
        loss_pi.backward()
        pi_optimizer.step()

        for p in q_params:
            p.requires_grad = True

        logger.store(LossPi=loss_pi.item(), **pi_info)

    def get_action(obs, deterministic=False):
        obs = torch.tensor(obs, device=device)
        return ac.act(obs, deterministic)

    def test_agent():
        for _ in range(num_test_episodes):
            obs, _ = test_env.reset()
            ep_ret, ep_len = 0, 0
            terminated, truncated = False, False
            while not (terminated or truncated):
                action = get_action(obs, True)
                obs, reward, terminated, truncated, _ = test_env.step(action)
                ep_ret += reward
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Main interaction loop
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    obs, _ = env.reset()
    ep_ret, ep_len = 0, 0

    logger.setup_pytorch_saver(ac)

    for t in range(total_steps):
        if t >= start_steps:
            action = get_action(obs)
        else:
            action = env.action_space.sample()

        obs2, reward, terminated, truncated, _ = env.step(action)
        ep_ret += reward
        ep_len += 1

        # Store if episode ended. However, if terminated only because of truncation,
        # we don't store the done transition because it may continue.
        replay_buffer.store(obs, action, reward, obs2, terminated and not truncated)
        obs = obs2

        if terminated or truncated:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, _ = env.reset()
            ep_ret, ep_len = 0, 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(batch)

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({"env": env}, None)

            test_agent()
            logger.set_step(t)
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("EpRet", with_min_and_max=True)
            logger.log_tabular("TestEpRet", with_min_and_max=True)
            logger.log_tabular("EpLen", average_only=True)
            logger.log_tabular("TestEpLen", average_only=True)
            logger.log_tabular("TotalEnvInteracts", t)
            logger.log_tabular("Q1Vals", with_min_and_max=True)
            logger.log_tabular("Q2Vals", with_min_and_max=True)
            logger.log_tabular("LogPi", with_min_and_max=True)
            logger.log_tabular("LossPi", average_only=True)
            logger.log_tabular("LossQ", average_only=True)
            logger.log_tabular("FPS", t / (time.time() - start_time))
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hockey-v0")
    parser.add_argument("--ac_kwargs", type=dict, default=dict())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps_per_epoch", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--replay_size", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_steps", type=int, default=1000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--update_every", type=int, default=10)
    parser.add_argument("--num_test_episodes", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    logger_kwargs = setup_logger_kwargs(args.env, args.seed)

    if args.env == "Hockey-v0":
        gym.register(
            id="Hockey-v0",
            entry_point="src.environment.environment:HockeyEnv",
        )

    cross_q(
        env_fn=lambda: gym.make(args.env),
        ac_kwargs=args.ac_kwargs,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        replay_size=args.replay_size,
        gamma=args.gamma,
        lr=args.lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        update_after=args.update_after,
        update_every=args.update_every,
        num_test_episodes=args.num_test_episodes,
        logger_kwargs=logger_kwargs,
        save_freq=args.save_freq,
        device=args.device,
    )
