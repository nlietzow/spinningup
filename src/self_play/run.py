import sys
from pathlib import Path

import wandb

sys.path.append(str(Path(__file__).parents[2]))

from src.self_play.pretraining import pretraining  # noqa: E402

if __name__ == "__main__":
    run = wandb.init(
        project="rl-challenge-self-play",
        sync_tensorboard=True,
        settings=wandb.Settings(silent=True),
    )
    success = False
    try:
        pretraining(run.id)
        success = True
    finally:
        run.finish(exit_code=int(not success))
