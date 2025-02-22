import sys
from pathlib import Path
from typing import NamedTuple

import wandb

sys.path.append(str(Path(__file__).parent.resolve()))

from src.algos.core.base import Base  # noqa: E402
from src.algos.cross_q.cross_q import CrossQ  # noqa: E402
from src.algos.sac.sac import SAC  # noqa: E402


class ModelMapping(NamedTuple):
    CROSS_Q: type[Base]
    SAC: type[Base]


MODELS = ModelMapping(
    CROSS_Q=CrossQ,
    SAC=SAC,
)

if __name__ == "__main__":
    from src.environment import make_hockey_env

    env = make_hockey_env()
    test_env = make_hockey_env()

    model = MODELS.SAC(env=env)
    run, error = wandb.init(project="cross_q"), None
    try:
        model.learn(
            total_steps=int(1e6),
            test_env=make_hockey_env(),
            wandb_run=run,
        )
    except (KeyboardInterrupt, Exception) as e:
        print(e)
        error = e
    finally:
        run.finish(exit_code=0 if error is None else 1)
