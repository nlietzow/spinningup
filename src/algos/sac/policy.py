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

        yield nn.Linear(sizes[-2], sizes[-1])
        yield output_activation()

    return nn.Sequential(*build())


class SquashedGaussianMLPActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
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

        pi_distribution = Normal(mu, std)
        if deterministic:
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


class MLPQFunction(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        activation: type[nn.Module],
    ):
        super().__init__()
        self.q = mlp(
            sizes=(obs_dim + act_dim, *hidden_sizes, 1),
            activation=activation,
            output_activation=nn.Identity,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        q = self.q(torch.cat((obs, act), dim=-1))
        return torch.squeeze(q, -1)  # Remove last dim


class SACActorCritic(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        init_alpha: float,
        alpha_trainable: bool,
        actor_hidden_sizes: tuple[int, ...],
        critic_hidden_sizes: tuple[int, ...],
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
            MLPQFunction(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_sizes=critic_hidden_sizes,
                activation=activation,
            )
            for _ in range(2)
        )

        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(init_alpha)),
            requires_grad=alpha_trainable,
        )

    def act(self, obs: np.ndarray, deterministic: bool) -> np.ndarray:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.pi.device)
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic=deterministic, with_logprob=False)
            return a.cpu().numpy()
