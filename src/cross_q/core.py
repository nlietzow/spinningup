import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def mlp_bn(sizes, activation, output_activation=nn.Identity):
    """
    Builds an MLP where every hidden layer (except the last) has a BatchNorm layer.
    Note: For a batch of shape (batch_size, features), nn.BatchNorm1d is used.
    """
    layers = []
    for j in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        layers.append(nn.BatchNorm1d(sizes[j + 1]))
        layers.append(activation())
    # Final layer without BatchNorm and (typically) without nonlinearity.
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    layers.append(output_activation())
    return nn.Sequential(*layers)


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # Apply tanh correction (see SAC paper appendix for details)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi


class MLPQFunctionBN(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        """
        Builds a Q network that uses BatchNorm after each hidden layer.
        The network takes as input the concatenation of state and action.
        """
        super().__init__()
        self.q = mlp_bn([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        """Standard forward pass on one (s, a) pair."""
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Remove last dim

    def forward_joint(self, obs, act, next_obs, next_act):
        """
        Implements the joint forward pass as described in the paper.
        It concatenates the current and next (state, action) pairs so that
        the BatchNorm layers compute normalization statistics over the union.
        Then it splits the result back into q and next_q.
        """
        cat_obs = torch.cat([obs, next_obs], dim=0)
        cat_act = torch.cat([act, next_act], dim=0)
        cat_input = torch.cat([cat_obs, cat_act], dim=-1)
        cat_q = self.q(cat_input)
        # Assume the original batch size is the same for obs and next_obs.
        q, next_q = torch.split(cat_q, obs.shape[0], dim=0)
        return torch.squeeze(q, -1), torch.squeeze(next_q, -1)


# An actor-critic model that uses the BN critic.
class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden_sizes, activation, act_limit
        )
        # Replace the standard Q networks with our BN-equipped critic networks.
        self.q1 = MLPQFunctionBN(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunctionBN(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()


# Example usage in a critic loss (conceptual):
def critic_loss(q1_net, q2_net, policy, obs, act, rews, next_obs, gamma, alpha):
    # Sample next actions from the current policy.
    next_act, next_logpi = policy(next_obs)
    # Use the joint forward pass on each Q network.
    q1, next_q1 = q1_net.forward_joint(obs, act, next_obs, next_act)
    q2, next_q2 = q2_net.forward_joint(obs, act, next_obs, next_act)
    # Compute the target as the minimum over the two Q's (and apply the entropy correction).
    next_q = torch.min(next_q1, next_q2)
    target = rews + gamma * (next_q - alpha * next_logpi)
    # Critic loss is the MSE error for both Q functions.
    loss_q1 = F.mse_loss(q1, target.detach())
    loss_q2 = F.mse_loss(q2, target.detach())
    loss = loss_q1 + loss_q2
    return loss
