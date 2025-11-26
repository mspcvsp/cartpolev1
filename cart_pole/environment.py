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

class Storage(object):

    def __init__(self,
                 args,
                 envs):

        self.num_steps = args.num_steps
        self.num_envs = args.num_envs
        self.device = args.device

        self.reset(envs)

    def store(self,
              paramid,
              value):

        assert self.step < self.num_steps, "Storage is full"
        self.__dict__[paramid][self.step] = value

    def update_episode_info(self,
                            info):

        if all(key in info for key in ["episode", "_episode"]):

            select_row = info["_episode"]

            self.episode_info.loc[select_row, "cumulative_reward"] =\
                info["episode"]["r"][select_row]

            self.episode_info.loc[select_row, "episode_length"] =\
                info["episode"]["l"][select_row]

        self.step += 1

    def reset(self,
              envs):

        self.step = 0

        shape2d = (self.num_steps, self.num_envs) 
        shape3d = shape2d + envs.single_observation_space.shape
        
        self.obs = torch.zeros(shape3d).to(self.device)
        self.actions = torch.zeros(shape3d).to(self.device)

        self.logprobs = torch.zeros(shape2d).to(self.device)
        self.rewards = torch.zeros(shape2d).to(self.device)
        self.terminated = torch.zeros(shape2d).to(self.device)
        self.truncated = torch.zeros(shape2d).to(self.device)
        self.values = torch.zeros(shape2d).to(self.device)

        self.episode_info =\
            pd.DataFrame([{"cumulative_reward": 0, "episode_length": 0}] *\
                         self.num_envs)
