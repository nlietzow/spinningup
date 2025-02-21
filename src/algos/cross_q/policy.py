import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from src.algos.sac.policy import SquashedGaussianMLPActor


def mlp_bn(
    sizes: tuple[int, ...],
    batch_norm_eps: float,
    batch_norm_momentum: float,
    activation: type[nn.Module],
    output_activation: type[nn.Module],
):
    def build():
        for j in range(len(sizes) - 2):
            yield nn.Linear(sizes[j], sizes[j + 1])
            yield nn.BatchNorm1d(
                sizes[j + 1],
                eps=batch_norm_eps,
                momentum=batch_norm_momentum,
            )
            yield activation()

        yield nn.Linear(sizes[-2], sizes[-1])
        yield output_activation()

    return nn.Sequential(*build())


class MLPQFunctionBN(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        batch_norm_eps: float,
        batch_norm_momentum: float,
        hidden_sizes: tuple[int, ...],
        activation: type[nn.Module],
    ):
        super().__init__()
        self.q = mlp_bn(
            sizes=(obs_dim + act_dim, *hidden_sizes, 1),
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
            activation=activation,
            output_activation=nn.Identity,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
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


class CrossQActorCritic(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        init_alpha: float,
        alpha_trainable: bool,
        batch_norm_eps: float,
        batch_norm_momentum: float,
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
            MLPQFunctionBN(
                obs_dim=obs_dim,
                act_dim=act_dim,
                batch_norm_eps=batch_norm_eps,
                batch_norm_momentum=batch_norm_momentum,
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
