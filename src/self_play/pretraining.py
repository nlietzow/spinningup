from sbx import CrossQ
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback

from src.environment import make_vec_hockey_env


def make_callback(eval_env, run_id):
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run_id}",
    )
    eval_callback = EvalCallback(eval_env=eval_env)
    return [wandb_callback, eval_callback]


def pretraining(run_id):
    env = make_vec_hockey_env(n_envs=4)
    eval_env = make_vec_hockey_env(n_envs=1)

    callback = make_callback(eval_env, run_id)
    model = CrossQ(
        CrossQ.policy_aliases["MlpPolicy"],
        env,
        tensorboard_log=f"logs/{run_id}",
    )
    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save(f"models/{run_id}/final_model")

    env.close()
    eval_env.close()
