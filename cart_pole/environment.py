import gymnasium as gym
import pandas as pd
import torch


def make_env(gym_id,
             idx,
             capture_video,
             run_name):

    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


def make_sync_vector_env(args):

    envs = gym.vector.SyncVectorEnv(
            [make_env(args.gym_id,
                      i,
                      args.capture_video,
                      args.run_name) for i in range(args.num_envs)]
        )
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete),\
        "only discrete action space is supported"

    return envs
