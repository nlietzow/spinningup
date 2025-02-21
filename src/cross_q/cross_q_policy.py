import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions.normal import Normal
from torch.nn.functional import softplus

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(
    sizes: tuple[int, ...],
    activation: type[nn.Module],
    output_activation: type[nn.Module],
):
    def build():
        for j in range(len(sizes) - 2):
            yield nn.Linear(sizes[j], sizes[j + 1])
            yield activation()

        # Final layer without non-linearity.
        yield nn.Linear(sizes[-2], sizes[-1])
        yield output_activation()

    return nn.Sequential(*build())


def mlp_bn(
    sizes: tuple[int, ...],
    activation: type[nn.Module],
    output_activation: type[nn.Module],
):
    def build():
        for j in range(len(sizes) - 2):
            yield nn.Linear(sizes[j], sizes[j + 1])
            yield nn.BatchNorm1d(
                sizes[j + 1], eps=0.001, momentum=0.01
            )  # todo: double check with the paper for eps and momentum
            yield activation()

        # Final layer without non-linearity.
        yield nn.Linear(sizes[-2], sizes[-1])
        yield output_activation()

    return nn.Sequential(*build())


class SquashedGaussianMLPActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, int],
        activation: type[nn.Module],
        act_limit: float,
    ):
        super().__init__()
        self.net = mlp(
            sizes=(obs_dim, *hidden_sizes),
            activation=activation,
            output_activation=activation,
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    @property
    def device(self) -> torch.device:
        return next(self.net.parameters()).device

    def forward(self, obs, deterministic: bool = False, with_logprob: bool = True):
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
            log_p_pi = pi_distribution.log_prob(pi_action).sum(dim=-1) - (
                2 * (np.log(2) - pi_action - softplus(-2 * pi_action))
            ).sum(dim=1)
        else:
            log_p_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, log_p_pi


class MLPQFunctionBN(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, int],
        activation: type[nn.Module],
    ):
        """
        Builds a Q network that uses BatchNorm after each hidden layer.
        The network takes as input the concatenation of state and action.
        """
        super().__init__()
        self.q = mlp_bn(
            sizes=(obs_dim + act_dim, *hidden_sizes, 1),
            activation=activation,
            output_activation=nn.Identity,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        """Standard forward pass on one (s, a) pair."""
        q = self.q(torch.cat((obs, act), dim=-1))
        return torch.squeeze(q, -1)  # Remove last dim

    def forward_joint(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
    ):
        """
        Implements the joint forward pass as described in the paper.
        It concatenates the current and next (state, action) pairs so that
        the BatchNorm layers compute normalization statistics over the union.
        Then it splits the result back into q and next_q.
        """
        cat_obs = torch.cat((obs, next_obs), dim=0)
        cat_act = torch.cat((act, next_act), dim=0)
        cat_input = torch.cat((cat_obs, cat_act), dim=-1)
        cat_q = self.q(cat_input)
        q, next_q = torch.split(cat_q, obs.shape[0], dim=0)

        return torch.squeeze(q, -1), torch.squeeze(next_q, -1)


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        alpha: float = 0.0,
        actor_hidden_sizes: tuple[int, int] = (256, 256),
        critic_hidden_sizes: tuple[int, int] = (1024, 1024),
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.pi = SquashedGaussianMLPActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=actor_hidden_sizes,
            activation=activation,
            act_limit=act_limit,
        )

        self.q1, self.q2 = (
            MLPQFunctionBN(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_sizes=critic_hidden_sizes,
                activation=activation,
            )
            for _ in range(2)
        )

        self.log_alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def act(self, obs: np.ndarray, deterministic: bool) -> np.ndarray:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.pi.device)
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic=deterministic, with_logprob=False)
            return a.cpu().numpy()
