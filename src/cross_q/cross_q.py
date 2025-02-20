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

    def to_tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def store(self, obs, act, rew, obs2, done):
        self.obs_buf[self.ptr] = self.to_tensor(obs)
        self.obs2_buf[self.ptr] = self.to_tensor(obs2)
        self.act_buf[self.ptr] = self.to_tensor(act)
        self.rew_buf[self.ptr] = self.to_tensor(rew)
        self.done_buf[self.ptr] = self.to_tensor(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size) -> Batch:
        idxs = torch.randint(
            0,
            self.size,
            (batch_size,),
            device=self.device,
        )
        return Batch(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            reward=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )


def cross_q(
    env_fn,
    ac_kwargs,
    seed,
    steps_per_epoch,
    epochs,
    replay_size,
    gamma,
    lr,
    alpha,
    batch_size,
    start_steps,
    update_after,
    update_every,
    num_test_episodes,
    logger_kwargs,
    save_freq,
    device,
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

    # Setup adaptive alpha parameter
    target_entropy = -float(act_dim)  # commonly set to -|A|
    log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
    alpha_optimizer = Adam([log_alpha], lr=lr, betas=(0.5, 0.999))

    # Create actor-critic module
    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    ac.to(device)

    # List of parameters for both Q-networks
    q_params = tuple(ac.q1.parameters()) + tuple(ac.q2.parameters())

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

        # Compute the target backup without gradients using adaptive alpha
        with torch.no_grad():
            q_pi = torch.min(q1_next, q2_next)
            backup = batch.reward + gamma * (1 - batch.done) * (
                q_pi - log_alpha.exp() * logp_a2
            )

        loss_q1 = ((q1_current - backup) ** 2).mean()
        loss_q2 = ((q2_current - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Set up function for computing CrossQ pi loss using adaptive alpha.
    # Returns both the loss and the log-probabilities for alpha update.
    def compute_loss_pi(batch: Batch):
        pi, logp_pi = ac.pi(batch.obs)
        q1_pi = ac.q1(batch.obs, pi)
        q2_pi = ac.q2(batch.obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (log_alpha.exp() * logp_pi - q_pi).mean()

        return loss_pi, logp_pi

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr, betas=(0.5, 0.999))
    q_optimizer = Adam(q_params, lr=lr, betas=(0.5, 0.999))

    def update(batch: Batch):
        # First update the Q-networks using the joint forward pass
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(batch)
        loss_q.backward()
        q_optimizer.step()
        logger.store(LossQ=loss_q.item())

        # Freeze Q-networks while updating the policy
        for p in q_params:
            p.requires_grad = False

        pi_optimizer.zero_grad()
        loss_pi, logp_pi = compute_loss_pi(batch)
        loss_pi.backward()
        pi_optimizer.step()

        for p in q_params:
            p.requires_grad = True

        logger.store(LossPi=loss_pi.item())

        # Adaptive alpha update
        alpha_optimizer.zero_grad()
        loss_alpha = -(log_alpha * (logp_pi.detach() + target_entropy)).mean()
        loss_alpha.backward()
        alpha_optimizer.step()
        logger.store(Alpha=log_alpha.exp().item(), LossAlpha=loss_alpha.item())

    def get_action(obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        action = ac.act(obs, deterministic)
        return action

    def test_agent():
        returns, lengths = np.zeros(num_test_episodes), np.zeros(num_test_episodes)
        for ep in range(num_test_episodes):
            obs, _ = test_env.reset()
            ep_ret, ep_len = 0, 0
            terminated, truncated = False, False
            while not (terminated or truncated):
                action = get_action(obs, True)
                obs, reward, terminated, truncated, _ = test_env.step(action)
                ep_ret += reward
                ep_len += 1

            returns[ep] = ep_ret
            lengths[ep] = ep_len

        logger.store(TestEpRet=returns.mean(), TestEpLen=lengths.mean())

    # Main interaction loop
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    obs, _ = env.reset()
    ep_ret, ep_len = 0, 0

    logger.setup_pytorch_saver(ac)

    for t in range(total_steps):
        logger.set_step(t)

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

            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("EpRet", with_min_and_max=True)
            logger.log_tabular("TestEpRet", with_min_and_max=True)
            logger.log_tabular("EpLen", average_only=True)
            logger.log_tabular("TestEpLen", average_only=True)
            logger.log_tabular("TotalEnvInteracts", t)
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
    parser.add_argument("--steps_per_epoch", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--replay_size", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_steps", type=int, default=1000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--num_test_episodes", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--logger_kwargs", type=dict, default=dict())

    args = parser.parse_args()

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
        logger_kwargs=args.logger_kwargs,
        save_freq=args.save_freq,
        device=args.device,
    )
