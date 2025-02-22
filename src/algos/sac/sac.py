from copy import deepcopy

import gymnasium as gym
import torch

from src.algos.common.replay_buffer import Batch
from src.algos.sac.policy import SACActorCritic
from src.algos.sac.sac_base import Base


class SAC(Base):
    actor_critic_class = SACActorCritic

    def __init__(
            self,
            env: gym.Env,
            replay_size: int = int(1e6),
            polyak: float = 0.995,
            init_alpha: float = 0.01,
            alpha_trainable: bool = True,
            actor_hidden_sizes: tuple[int, ...] = (256, 256),
            critic_hidden_sizes: tuple[int, ...] = (256, 256),
            batch_size: int = 256,
            gamma: float = 0.99,
            betas: tuple[float, float] = (0.9, 0.999),
            lr: float = 3e-4,
            device: str = "auto",
    ):
        super().__init__(
            env=env,
            replay_size=replay_size,
            init_alpha=init_alpha,
            alpha_trainable=alpha_trainable,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
            batch_size=batch_size,
            gamma=gamma,
            betas=betas,
            lr=lr,
            policy_delay=1,
            batch_norm_eps=None,
            batch_norm_momentum=None,
            device=device,
        )
        self.ac.cpu()
        self.ac_target = deepcopy(self.ac)
        self.ac.to(self.device)
        self.ac_target.to(self.device)

        for p in self.ac_target.parameters():
            p.requires_grad = False

        self.polyak = polyak

    def compute_loss_q(self, batch: Batch) -> torch.Tensor:
        q1 = self.ac.q1(batch.obs, batch.act)
        q2 = self.ac.q2(batch.obs, batch.act)

        with torch.no_grad():
            a2, log_p_a2 = self.ac.pi(batch.obs2)
            q1_target = self.ac_target.q1(batch.obs2, a2)
            q2_target = self.ac_target.q2(batch.obs2, a2)
            q_target = torch.min(q1_target, q2_target)
            backup = batch.reward + self.gamma * (1 - batch.done) * (
                    q_target - self.ac.log_alpha.exp() * log_p_a2
            )

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def update(self, batch: Batch, **kwargs):
        super().update(batch=batch, update_policy=True)

        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
