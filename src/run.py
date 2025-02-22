import sys
from pathlib import Path
from typing import NamedTuple

import wandb

from src.algos.cross_q.cross_q_self_play import CrossQSelfPlay

sys.path.append(str(Path(__file__).parents[1].resolve()))

from src.environment import make_hockey_env  # noqa: E402
from src.algos.cross_q.cross_q import CrossQ  # noqa: E402
from src.algos.sac.sac import SAC  # noqa: E402


class ModelMapping(NamedTuple):
    SAC: type[SAC]
    CROSS_Q: type[CrossQ]
    Cross_Q_SELF_PLAY: type[CrossQSelfPlay]


MODELS = ModelMapping(SAC=SAC, CROSS_Q=CrossQ, Cross_Q_SELF_PLAY=CrossQSelfPlay)

if __name__ == "__main__":
    run = wandb.init(
        project="cross_q",
        settings=wandb.Settings(silent=True),
    )

    model, env, test_env = None, None, None
    try:
        env = make_hockey_env()
        test_env = make_hockey_env()
        model = MODELS.CROSS_Q(env=env)
        model.learn(
            total_steps=5_000_000,
            test_env=test_env,
            wandb_run=run,
        )
    except KeyboardInterrupt:
        if model is not None:
            model.save_model(Path("models") / run.id / "final_model")
    finally:
        if env is not None:
            env.close()
        if test_env is not None:
            test_env.close()
        run.finish()
