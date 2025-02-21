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

import src.sac.core as core
from src.config import setup_logger_kwargs
from src.logx import EpochLogger


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


def sac(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
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
    Soft Actor-Critic (SAC)

    SAC is an off-policy actor-critic deep RL algorithm that optimizes a stochastic
    policy in an entropy-regularized reinforcement learning framework. The actor aims
    to maximize expected return while also maximizing entropy --- that is,
    succeed at the task while acting as randomly as possible.

    Args:
        env_fn : A function which creates a copy of the environment.
        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``     (batch, act_dim)  | Numpy array of actions for each
                                          | observation.
            ``pi``      N/A               | Torch Distribution object, containing
                                          | actions and log probs.
            ``q1``      (batch,)          | Tensor containing one current estimate
                                          | of Q* for the provided observations
                                          | and actions.
            ``q2``      (batch,)          | Tensor containing the other current
                                          | estimate of Q* for the provided
                                          | observations and actions.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        device (str): Device to run the model on.
    """
    local_vars = locals()
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(local_vars)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Create actor-critic module and target networks
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _device = torch.device(device)
    logger.log("Device: %s" % device)
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(_device)
    ac_targ = deepcopy(ac).to(_device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=_device
    )

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log("\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        obs, action, reward, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        q1 = ac.q1(obs, action)
        q2 = ac.q2(obs, action)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(obs2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(obs2, a2)
            q2_pi_targ = ac_targ.q2(obs2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = reward + gamma * (1 - done) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(
            Q1Vals=q1.detach().cpu().numpy(),
            Q2Vals=q2.detach().cpu().numpy(),
        )

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        obs = data["obs"]
        pi, logp_pi = ac.pi(obs)
        q1_pi = ac.q1(obs, pi)
        q2_pi = ac.q2(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=_device)
        return ac.act(obs, deterministic)

    def test_agent():
        for _ in range(num_test_episodes):
            obs, _ = test_env.reset()
            ep_ret, ep_len = 0, 0
            terminated, truncated = False, False

            while not (terminated or truncated):
                # Take deterministic actions at test time
                action = get_action(obs, True)
                obs, reward, terminated, truncated, _ = test_env.step(action)
                ep_ret += reward
                ep_len += 1

            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()

    obs, _ = env.reset()
    ep_ret, ep_len = 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t >= start_steps:
            action = get_action(obs)
        else:
            action = env.action_space.sample()

        # Step the env
        obs2, reward, terminated, truncated, _ = env.step(action)

        ep_ret += reward
        ep_len += 1

        # Store experience to replay buffer
        replay_buffer.store(obs, action, reward, obs2, terminated)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        obs = obs2

        # End of trajectory handling
        if terminated or truncated:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, _ = env.reset()
            ep_ret, ep_len = 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({"env": env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
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
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="sac")
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name, seed=args.seed)

    sac(
        lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
