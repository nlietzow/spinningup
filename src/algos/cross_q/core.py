import torch
from gymnasium import spaces

from src.algos.core.algorithm import ActorCriticBase
from src.algos.core.critic import CriticBase
from src.algos.core.utils import mlp_bn


class MLPQFunctionBN(CriticBase):
    critic_builder = staticmethod(mlp_bn)

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        batch_norm_eps: float,
        batch_norm_momentum: float,
    ):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )

    def forward_joint(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
    ):
        cat_obs = torch.cat((obs, next_obs), dim=0)
        cat_act = torch.cat((act, next_act), dim=0)
        cat_input = torch.cat((cat_obs, cat_act), dim=-1)
        cat_q = self.q(cat_input)
        q, next_q = torch.split(cat_q, obs.shape[0], dim=0)

        return torch.squeeze(q, -1), torch.squeeze(next_q, -1)


class CrossQActorCritic(ActorCriticBase):
    critic_class = MLPQFunctionBN

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        init_alpha: float,
        alpha_trainable: bool,
        actor_hidden_sizes: tuple[int, ...],
        critic_hidden_sizes: tuple[int, ...],
        batch_norm_eps: float,
        batch_norm_momentum: float,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            init_alpha=init_alpha,
            alpha_trainable=alpha_trainable,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )
