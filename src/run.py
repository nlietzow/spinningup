import sys
from pathlib import Path
from typing import NamedTuple

import wandb

sys.path.append(str(Path(__file__).parent.resolve()))

from src.algos.cross_q.cross_q import CrossQ  # noqa: E402
from src.algos.sac.sac import SAC  # noqa: E402


class ModelMapping(NamedTuple):
    SAC: type[SAC]
    CROSS_Q: type[CrossQ]


MODELS = ModelMapping(SAC=SAC, CROSS_Q=CrossQ)

if __name__ == "__main__":
    from src.environment import make_hockey_env

    env = make_hockey_env()
    test_env = make_hockey_env()

    model = MODELS.SAC(env=env)
    run, success = wandb.init(project="cross_q"), False
    try:
        model.learn(
            total_steps=int(1e6),
            test_env=make_hockey_env(),
            wandb_run=run,
        )
        success = True
    finally:
        run.finish(exit_code=0 if success else 1)
