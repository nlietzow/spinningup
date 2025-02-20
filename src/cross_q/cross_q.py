import itertools
import sys
import time
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.optim import Adam

sys.path.append(str(Path(__file__).parents[2].resolve()))

import src.cross_q.core as core
from src.config import setup_logger_kwargs
from src.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents with GPU storage.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        # Initialize tensors directly on the specified device
        self.obs_buf = torch.zeros(
            core.combined_shape(size, obs_dim), dtype=torch.float32, device=device
        )
        self.obs2_buf = torch.zeros(
            core.combined_shape(size, obs_dim), dtype=torch.float32, device=device
        )
        self.act_buf = torch.zeros(
            core.combined_shape(size, act_dim), dtype=torch.float32, device=device
        )
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        # Convert numpy arrays to tensors and move to device
        self.obs_buf[self.ptr] = torch.as_tensor(obs, device=self.device)
        self.obs2_buf[self.ptr] = torch.as_tensor(next_obs, device=self.device)
        self.act_buf[self.ptr] = torch.as_tensor(act, device=self.device)
        self.rew_buf[self.ptr] = torch.as_tensor(rew, device=self.device)
        self.done_buf[self.ptr] = torch.as_tensor(done, device=self.device)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )


def cross_q(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    lr=1e-3,
    alpha=0.2,
    batch_size=256,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    logger_kwargs=dict(),
    save_freq=1,
    device="auto",
):
    """
    Soft Actor-Critic (SAC) with a BN-equipped critic that uses a joint forward pass.
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

    # Create actor-critic module
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _device = torch.device(device)
    logger.log("Device: %s" % device)
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(_device)

    # List of parameters for both Q-networks
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=_device
    )

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log("\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts)

    # Set up function for computing SAC Q-losses with joint forward pass
    def compute_loss_q(data):
        obs, action, reward, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        with torch.no_grad():
            # Get next actions and corresponding log-probs
            a2, logp_a2 = ac.pi(obs2)
            # Use the joint forward pass so that BatchNorm sees both current and next samples
            q1_current, q1_next = ac.q1.forward_joint(obs, action, obs2, a2)
            q2_current, q2_next = ac.q2.forward_joint(obs, action, obs2, a2)
            # Use the minimum of the next Q estimates for the target, with entropy correction
            q_pi = torch.min(q1_next, q2_next)
            backup = reward + gamma * (1 - done) * (q_pi - alpha * logp_a2)

        loss_q1 = ((q1_current - backup) ** 2).mean()
        loss_q2 = ((q2_current - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(
            Q1Vals=q1_current.detach().cpu().numpy(),
            Q2Vals=q2_current.detach().cpu().numpy(),
        )

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        obs = data["obs"]
        pi, logp_pi = ac.pi(obs)
        q1_pi = ac.q1(obs, pi)
        q2_pi = ac.q2(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (alpha * logp_pi - q_pi).mean()

        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())
        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr, betas=(0.5, 0.999))
    q_optimizer = Adam(q_params, lr=lr, betas=(0.5, 0.999))

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First update the Q-networks using the joint forward pass
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks while updating the policy
        for p in q_params:
            p.requires_grad = False

        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        for p in q_params:
            p.requires_grad = True

        logger.store(LossPi=loss_pi.item(), **pi_info)

    def get_action(obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=_device)
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

    for t in range(total_steps):
        if t >= start_steps:
            action = get_action(obs)
        else:
            action = env.action_space.sample()

        obs2, reward, terminated, truncated, _ = env.step(action)
        ep_ret += reward
        ep_len += 1

        replay_buffer.store(obs, action, reward, obs2, terminated)
        obs = obs2

        if terminated or truncated:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, _ = env.reset()
            ep_ret, ep_len = 0, 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

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
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="cross_q")
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name, seed=args.seed)

    cross_q(
        lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,  # make sure this is the BN-enabled version
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
