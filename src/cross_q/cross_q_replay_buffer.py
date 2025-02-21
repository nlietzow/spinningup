from typing import NamedTuple, Optional, SupportsFloat, Union

import numpy as np
import torch


class Batch(NamedTuple):
    obs: torch.Tensor
    act: torch.Tensor
    reward: torch.Tensor
    obs2: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for Cross Q-learning agents with GPU storage.
    """

    def __init__(
        self,
        obs_dim: tuple[int, ...],
        act_dim: int,
        size: int,
        device: torch.device,
    ):
        # Initialize tensors directly on the specified device
        self.obs_buf, self.obs2_buf = (
            torch.zeros(
                self.combined_shape(size, obs_dim),
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        )
        self.act_buf = torch.zeros(
            self.combined_shape(size, act_dim),
            dtype=torch.float32,
            device=device,
        )
        self.rew_buf, self.done_buf = (
            torch.zeros(size, dtype=torch.float32, device=device) for _ in range(2)
        )

        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: SupportsFloat,
        obs2: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = self.to_tensor(obs)
        self.obs2_buf[self.ptr] = self.to_tensor(obs2)
        self.act_buf[self.ptr] = self.to_tensor(act)
        self.rew_buf[self.ptr] = self.to_tensor(np.array([rew], dtype=np.float32))
        self.done_buf[self.ptr] = self.to_tensor(np.array([done], dtype=np.float32))
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

    @staticmethod
    def combined_shape(
        length: int, shape: Optional[Union[tuple[int, ...], int]]
    ) -> tuple[int, ...]:
        if shape is None:
            return (length,)
        if isinstance(shape, int):
            return (length, shape)

        return (length, *shape)
