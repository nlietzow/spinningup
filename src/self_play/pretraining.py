import multiprocessing as mp

from sbx import CrossQ
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    EveryNTimesteps,
)
from wandb.integration.sb3 import WandbCallback

from src.environment import make_vec_hockey_env


def make_callback(eval_env, run_id):
    wandb_callback = WandbCallback(model_save_path=f"models/{run_id}")
    eval_callback = EvalCallback(
        eval_env=eval_env, best_model_save_path=f"models/{run_id}"
    )
    _checkpoint_callback = CheckpointCallback(save_freq=1, save_path=f"models/{run_id}")
    every_n_timesteps_callback = EveryNTimesteps(
        n_steps=1_000_000, callback=_checkpoint_callback
    )
    return [
        wandb_callback,
        eval_callback,
        every_n_timesteps_callback,
    ]


def main(run_id):
    env = None
    eval_env = None
    model = None
    try:
        env = make_vec_hockey_env(n_envs=mp.cpu_count() - 1)
        eval_env = make_vec_hockey_env(n_envs=1)

        print(f"Created env with {env.num_envs} processes.")

        callback = make_callback(eval_env, run_id)
        model = CrossQ(
            CrossQ.policy_aliases["MlpPolicy"],
            env,
            verbose=0,
            tensorboard_log=f"logs/{run_id}",
        )
        model.learn(total_timesteps=500_000, callback=callback)

    finally:
        if model is not None:
            model.save(f"models/{run_id}/final_model")
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()
