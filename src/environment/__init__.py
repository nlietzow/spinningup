import gymnasium as gym


def make_hockey_env(weak: bool = False) -> gym.Env:
    # check if the env is registered
    if "Hockey-v0" not in gym.envs.registry:
        gym.register(
            id="Hockey-v0",
            entry_point="src.environment.environment:HockeyEnv",
        )
    return gym.make("Hockey-v0", weak=weak)
