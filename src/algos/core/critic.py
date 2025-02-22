from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
import torch.nn as nn


class CriticBase(nn.Module, ABC):
    @property
    @abstractmethod
    def critic_builder(self) -> Callable[..., nn.Sequential]:
        pass

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_sizes: tuple[int, ...],
            batch_norm_eps: Optional[float],
            batch_norm_momentum: Optional[float],
    ):
        super().__init__()
        kwargs = {}
        if batch_norm_eps is not None:
            kwargs["batch_norm_eps"] = batch_norm_eps
        if batch_norm_momentum is not None:
            kwargs["batch_norm_momentum"] = batch_norm_momentum

        sizes: tuple[int, ...] = (obs_dim + act_dim, *hidden_sizes, 1)
        self.q = self.critic_builder(
            sizes=sizes,
            activation=nn.ReLU,
            output_activation=nn.Identity,
            **kwargs,
        )
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum

    @property
    def hidden_sizes(self) -> tuple[int, ...]:
        return tuple(
            layer.out_features for layer in self.q if isinstance(layer, nn.Linear)
        )[:-1]

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = self.q(torch.cat((obs, act), dim=-1))
        return torch.squeeze(x, -1)  # Remove last dim
