import wandb
from sbx import CrossQ
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    EveryNTimesteps,
)
from src.environment import make_vec_hockey_env
from wandb.integration.sb3 import WandbCallback

from src.self_play.update_player2_callback import SelfPlayCallback


def make_callback(eval_env, run_id):
    wandb_callback = WandbCallback(model_save_path=f"models/{run_id}")
    self_play_callback = SelfPlayCallback()
    eval_callback = EvalCallback(
        eval_env=eval_env, best_model_save_path=f"models/{run_id}"
    )
    _checkpoint_callback = CheckpointCallback(save_freq=1, save_path=f"models/{run_id}")
    every_n_timesteps_callback = EveryNTimesteps(
        n_steps=1_000_000, callback=_checkpoint_callback
    )
    return [
        wandb_callback,
        self_play_callback,
        eval_callback,
        every_n_timesteps_callback,
    ]


def main(run_id):
    env = None
    eval_env = None
    model = None
    try:
        env = make_vec_hockey_env(n_envs=4)
        eval_env = make_vec_hockey_env(n_envs=4)

        callback = make_callback(eval_env, run_id)
        model = CrossQ(
            CrossQ.policy_aliases["MlpPolicy"],
            env,
            verbose=0,
            stats_window_size=1,
            tensorboard_log=f"logs/{run_id}",
        )
        model.learn(total_timesteps=20_000_000, callback=callback)

    finally:
        if model is not None:
            model.save(f"models/{run_id}/final_model")
        if env is not None:
            env.close()
        if eval_env is not None:
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
