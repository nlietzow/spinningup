import sys
from pathlib import Path

import wandb

sys.path.append(str(Path(__file__).parent.resolve()))

from src.algos.cross_q.cross_q import CrossQ  # noqa: E402

if __name__ == "__main__":
    from src.environment import make_hockey_env

    model = CrossQ(env=make_hockey_env())
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
