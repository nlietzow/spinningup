import sys
from pathlib import Path
from typing import NamedTuple

import wandb

sys.path.append(str(Path(__file__).parents[1].resolve()))

from src.environment import make_hockey_env  # noqa: E402
from src.algos.cross_q.cross_q import CrossQ  # noqa: E402
from src.algos.sac.sac import SAC  # noqa: E402


class ModelMapping(NamedTuple):
    SAC: type[SAC]
    CROSS_Q: type[CrossQ]


MODELS = ModelMapping(SAC=SAC, CROSS_Q=CrossQ)

if __name__ == "__main__":
    for model_cls, alpha, alpha_trainable in (
        (MODELS.SAC, 1 / 100000, False),
        (MODELS.SAC, 1 / 1000, False),
        (MODELS.SAC, 1 / 10, False),
        (MODELS.SAC, 1 / 5, False),
        (MODELS.SAC, 1 / 5, True),
        (MODELS.CROSS_Q, 1 / 5, True),
    ):
        run = wandb.init(
            project="cross_q",
            settings=wandb.Settings(silent=True),
        )
        env = make_hockey_env(weak=True)
        test_env = make_hockey_env(weak=True)
        model = model_cls(
            env=env,
            init_alpha=alpha,
            alpha_trainable=alpha_trainable,
        )
        model.learn(
            total_steps=750_000,
            test_env=test_env,
            wandb_run=run,
        )
        model.save_model(Path("models") / run.id / "final_model")
        env.close()
        test_env.close()
        run.finish()
