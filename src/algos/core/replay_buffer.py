from pathlib import Path
from typing import NamedTuple, Optional, SupportsFloat, Union

import numpy as np
import torch
from tensordict import TensorDict


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
        max_size: int,
        device: torch.device,
    ):
        # Initialize tensors directly on the specified device
        self.obs_buf, self.obs2_buf = (
            torch.zeros(
                self.combined_shape(max_size, obs_dim),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                self.combined_shape(max_size, obs_dim),
                dtype=torch.float32,
                device=device,
            ),
        )
        self.act_buf = torch.zeros(
            self.combined_shape(max_size, act_dim),
            dtype=torch.float32,
            device=device,
        )
        self.rew_buf, self.done_buf = (
            torch.zeros(max_size, dtype=torch.float32, device=device) for _ in range(2)
        )

        self.ptr, self.size = 0, 0
        self.device = device

    @property
    def max_size(self) -> int:
        return self.obs_buf.shape[0]

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

    def sample_batch(self, batch_size: int) -> Batch:
        indices = torch.randint(
            0,
            self.size,
            (batch_size,),
            device=self.device,
        )
        return Batch(
            obs=self.obs_buf[indices],
            obs2=self.obs2_buf[indices],
            act=self.act_buf[indices],
            reward=self.rew_buf[indices],
            done=self.done_buf[indices],
        )

    @staticmethod
    def combined_shape(
        length: int, shape: Optional[Union[tuple[int, ...], int]]
    ) -> tuple[int, ...]:
        if shape is None:
            return (length,)
        if isinstance(shape, int):
            return length, shape

        return length, *shape

    def save(self, path: Path) -> None:
        td = TensorDict(
            {
                "obs_buf": self.obs_buf.detach().cpu(),
                "obs2_buf": self.obs2_buf.detach().cpu(),
                "act_buf": self.act_buf.detach().cpu(),
                "rew_buf": self.rew_buf.detach().cpu(),
                "done_buf": self.done_buf.detach().cpu(),
                "ptr": torch.tensor(self.ptr, device="cpu"),
                "size": torch.tensor(self.size, device="cpu"),
                "max_size": torch.tensor(self.max_size, device="cpu"),
            },
            batch_size=[self.max_size],
        )
        torch.save(td, path)

    def load(self, path: Path) -> None:
        data = torch.load(path)

        if data["max_size"].item() != self.max_size:
            raise ValueError(
                f"Max size mismatch: {data['max_size'].item()} != {self.max_size}"
            )

        self.obs_buf = data["obs_buf"].to(self.device)
        self.obs2_buf = data["obs2_buf"].to(self.device)
        self.act_buf = data["act_buf"].to(self.device)
        self.rew_buf = data["rew_buf"].to(self.device)
        self.done_buf = data["done_buf"].to(self.device)
        self.ptr = data["ptr"].item()
        self.size = data["size"].item()
