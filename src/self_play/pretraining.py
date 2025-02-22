import multiprocessing as mp
import sys
from pathlib import Path

import wandb
from sbx import CrossQ
from stable_baselines3.common.callbacks import (
    EvalCallback,
)
from wandb.integration.sb3 import WandbCallback

sys.path.append(str(Path(__file__).parents[2]))

from src.environment import make_vec_hockey_env


def make_callback(eval_env):
    wandb_callback = WandbCallback()
    eval_callback = EvalCallback(eval_env=eval_env)
    return [wandb_callback, eval_callback]


def main(run_id):
    env = make_vec_hockey_env(n_envs=max(1, mp.cpu_count() // 2))
    eval_env = make_vec_hockey_env(n_envs=1)

    print(f"Created env with {env.num_envs} processes.")

    callback = make_callback(eval_env)
    model = CrossQ(
        CrossQ.policy_aliases["MlpPolicy"],
        env,
        verbose=0,
        tensorboard_log=f"logs/{run_id}",
    )
    model.learn(total_timesteps=500_000, callback=callback)

    model.save(f"models/{run_id}/final_model")
    model.save_replay_buffer(f"models/{run_id}/replay_buffer")
    wandb.save(f"models/{run_id}/final_model", base_path="models")
    wandb.save(f"models/{run_id}/replay_buffer", base_path="models")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    run = wandb.init(
        project="rl-challenge-self-play",
        sync_tensorboard=True,
        settings=wandb.Settings(silent=True),
    )
    success = False
    try:
        main(run.id)
        success = True
    finally:
        run.finish(exit_code=int(not success))
