import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class Agent(nn.Module):

    def __init__(self,
                 envs):

        super(Agent, self).__init__()

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


class Storage(object):

    def __init__(self,
                 args,
                 envs):

        self.num_steps = args.num_steps
        self.num_envs = args.num_envs
        self.device = args.device

        self.reset(envs)

    def store_state_info(self,
                         obs,
                         terminated,
                         truncated,
                         info):

        if self.step is None:
            self.step = 0
        else:
            self.step += 1

        assert self.step < self.num_steps, "Storage is full"

        self.obs[self.step, :, :] = obs
        self.terminated[self.step, :] = terminated
        self.truncated[self.step, :] = truncated

        if len(info) > 0:

            key0 = "episode"
            epr_mask = info[key0]["_r"]
            epr_values = info[key0]["r"][epr_mask].astype('float32')
            
            self.episodic_rewards[self.step, epr_mask] =\
                torch.tensor(epr_values).to(self.device)

            pdb.set_trace()

    def store_action_info(self,
                          values,
                          actions,
                          logprobs):

        self.values[self.step, :] = values
        self.actions[self.step, :] = actions
        self.logprobs[self.step, :] = logprobs

    def store_rewards(self,
                      rewards):

        self.rewards[self.step, :] = rewards

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
