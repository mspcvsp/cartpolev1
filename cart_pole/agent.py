"""
"The 37 Implementation Details of Proximal Policy Optimization" 25 March 2022
Huang, Shengyi; Dossa, Rousslan Fernand Julien; Raffin, Antonin; Kanervisto,
Anssi; Wang, Weixun

https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
https://github.com/vwxyzjn/ppo-implementation-details
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from .utils import init_rngs


class MinMaxScaler(object):

    def __init__(self,
                 args,
                 envs):

        obs_space = copy.deepcopy(envs.single_observation_space)

        self.is_bounded =\
            np.logical_and(obs_space.bounded_above,
                           obs_space.bounded_below)

        self.is_unbounded =\
            np.logical_not(self.is_bounded)

        self.is_bounded = torch.tensor(self.is_bounded)
        self.is_unbounded = torch.tensor(self.is_unbounded)
        
        self.low = torch.tensor(obs_space.low)
        self.inv_range = torch.ones_like(self.low)

        self.inv_range[self.is_bounded] =\
            1 / (torch.tensor(obs_space.high[self.is_bounded]) -\
                 self.low[self.is_bounded])

        self.inv_range = self.inv_range.to(args.device)
        self.low = self.low.to(args.device)

    def __call__(self,
                 x):

        scaled_x = x.clone()
        
        target_shape = torch.tensor(scaled_x.shape)
        expand_shape = [1] *(len(target_shape) - 1) + [-1]
        
        low_nd = self.low.view(*expand_shape).expand(*target_shape)

        inv_range_nd =\
            self.inv_range.view(*expand_shape).expand(*target_shape)
        
        is_unbounded_nd =\
            self.is_unbounded.view(*expand_shape).expand(*target_shape)
        
        is_bounded_nd =\
            self.is_bounded.view(*expand_shape).expand(*target_shape)

        scaled_x[is_bounded_nd] =\
            (scaled_x[is_bounded_nd] - low_nd[is_bounded_nd]) *\
            inv_range_nd[is_bounded_nd]
        
        scaled_x[is_unbounded_nd] =\
            scaled_x[is_unbounded_nd] / (1 + scaled_x[is_unbounded_nd].abs())

        return scaled_x


class Agent(nn.Module):

    def __init__(self,
                 envs):

        super().__init__()

        obs_space_shape = envs.single_observation_space.shape

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):

        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def layer_init(layer,
               std=np.sqrt(2),
               bias_const=0.0):

    torch.nn.init.orthogonal_(layer.weight,
                              std)

    torch.nn.init.constant_(layer.bias,
                            bias_const)

    return layer


class PPOTrainer(object):

    def __init__(self,
                 args,
                 envs):

        self.num_steps = args.num_steps
        self.num_envs = args.num_envs
        self.device = args.device
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda

        self.rng = init_rngs(args)        

        self.obs_scaler = MinMaxScaler(args,
                                       envs)

        self.obs_space_shape = envs.single_observation_space.shape.copy()

        self.reset(envs)

    def reset(self,
              envs):

        self.step = None
        shape2d = (self.num_steps, self.num_envs) 
        shape3d = shape2d + envs.single_observation_space.shape

        self.obs = torch.zeros(shape3d).to(self.device)
        self.terminated = torch.zeros(shape2d).to(self.device)
        self.truncated = torch.zeros(shape2d).to(self.device)

        self.actions = torch.zeros(shape3d).to(self.device)
        self.logprobs = torch.zeros(shape2d).to(self.device)
        self.rewards = torch.zeros(shape2d).to(self.device)
        self.values = torch.zeros(shape2d).to(self.device)
        self.episodic_rewards = torch.zeros(shape2d).to(self.device)

        seeds = [int(self.rng.uniform(0, 1E9))
                 for _ in range(self.num_envs)]

        self.next_obs =\
            torch.Tensor(envs.reset(seed=seeds)[0]).to(self.device)

        self.next_obs = self.obs_scaler(self.next_obs)

        self.next_rewards =\
            torch.Tensor(np.array([0] * self.num_envs)).to(self.device)

        self.next_terminated =\
            torch.Tensor(np.array([False] * self.num_envs)).to(self.device)

        self.next_truncated =\
            torch.Tensor(np.array([False] * self.num_envs)).to(self.device)

        self.advantages = torch.zeros_like(self.rewards).to(self.device)

        self.next_info = {}
    
    def store_state_info(self):

        if self.step is None:
            self.step = 0
        else:
            self.step += 1

        assert self.step < self.num_steps, "Storage is full"

        self.obs[self.step, :, :] = self.next_obs.clone()
        self.rewards[self.step, :] = self.next_rewards.clone()
        self.terminated[self.step, :] = self.next_terminated.clone()
        self.truncated[self.step, :] = self.next_truncated.clone()

        if len(self.next_info) > 0:

            key0 = "episode"
            epr_mask = self.next_info[key0]["_r"]

            epr_values =\
                self.next_info[key0]["r"][epr_mask].astype('float32')
            
            self.episodic_rewards[self.step, epr_mask] =\
                torch.tensor(epr_values).to(self.device)

    def store_action_info(self,
                          values,
                          actions,
                          logprobs):

        self.values[self.step, :] = values
        self.actions[self.step, :] = actions
        self.logprobs[self.step, :] = logprobs

    def rollout(self,
                agent,
                envs):

        self.reset(envs)

        for _ in range(self.num_steps):

            self.store_state_info()

            with torch.no_grad():

                actions, logprobs, _, values =\
                    agent.get_action_and_value(self.next_obs)

                self.store_action_info(values.flatten(),
                                       actions,
                                       logprobs)
    
                (self.next_obs,
                 self.next_rewards,
                 self.next_terminated,
                 self.next_truncated,
                 self.next_info) = envs.step(actions.cpu().numpy())

                self.next_obs = torch.Tensor(self.next_obs).to(self.device)
                self.next_obs = self.obs_scaler(self.next_obs)

                self.next_rewards =\
                    torch.tensor(self.next_rewards).to(self.device).view(-1)
                
                self.next_terminated =\
                    torch.Tensor(self.next_terminated).to(self.device)
                
                self.next_truncated =\
                    torch.Tensor(self.next_truncated).to(self.device)

        self.compute_gae(agent)

    def flatten_batch_rollout(self):

        obs_space_shape = trainer.obs.shape[-1]

        self.obs = self.obs.reshape((-1,) + obs_space_shape)
        self.logprobs = logprobs.reshape(-1)

    def compute_gae(self,
                    agent):
        """
        - https://tianshou.org/en/stable/02_deep_dives/L4_GAE.html
        - https://notanymike.github.io/GAE/
        - https://mochan.org/posts/gae/
        """
        with torch.no_grad():

            gamma_lambda = self.gamma * self.gae_lambda

            for cur_step in reversed(range(self.num_steps)):

                if cur_step == self.num_steps - 1:

                    next_values =\
                        agent.get_value(self.next_obs).reshape(1, -1)

                    next_rewards = self.next_rewards
                    next_terminated = self.next_terminated
                    next_truncated = self.next_truncated
                    next_advantages = torch.zeros_like(next_values)
                #--------------------------------------------
                else:
                    next_step = cur_step + 1
    
                    next_values = self.values[next_step]
                    next_rewards = self.rewards[next_step]
                    next_terminated = self.terminated[next_step]
                    next_truncated = self.truncated[next_step]
                    next_advantages = self.advantages[next_step]

                reset_mask = (1 - next_terminated) * (1 - next_truncated)
    
                td_residuals =\
                     next_rewards -\
                     self.gamma * next_values * reset_mask -\
                     self.values[cur_step, :]

                self.advantages[cur_step] =\
                    td_residuals + gamma_lambda * next_advantages * reset_mask
