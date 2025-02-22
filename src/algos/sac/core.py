from gymnasium import spaces

from src.algos.core.algorithm import ActorCriticBase
from src.algos.core.critic import CriticBase
from src.algos.core.utils import mlp


class SACCritic(CriticBase):
    critic_builder = staticmethod(mlp)

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        batch_norm_eps: None = None,  # for compatibility with CrossQ
        batch_norm_momentum: None = None,  # for compatibility with CrossQ
    ):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )


class SACActorCritic(ActorCriticBase):
    critic_class = SACCritic

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        init_alpha: float,
        alpha_trainable: bool,
        actor_hidden_sizes: tuple[int, ...],
        critic_hidden_sizes: tuple[int, ...],
        batch_norm_eps: None = None,  # for compatibility with CrossQ
        batch_norm_momentum: None = None,  # for compatibility with CrossQ
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
