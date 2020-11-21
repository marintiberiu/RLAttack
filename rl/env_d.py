import os

import gym
import numpy as np
import torch
from torchvision.models import resnet18
from PIL import Image


class AttackEnv(gym.core.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
        self.actions = ['N', 'E', 'S', 'V', '+', '-', 'R', 'Q']
        self.action_space = gym.spaces.Discrete(8)
        self.image = None
        self.target_class = None
        self.net = resnet18(pretrained=False, num_classes=10).eval().cuda()
        self.net.load_state_dict(torch.load('saves/resnet18_cifar10.sv'))
        self.state = np.zeros((32, 32), dtype=np.float32)
        self.observation = np.zeros(10, dtype=np.float32)
        self.max_row = 32
        self.max_col = 32
        self.row = self.max_row // 2
        self.col = self.max_col // 2
        # logging tools
        self.episodes = 0
        self.successes = 0
        self.q_idx = 0
        self.n_queries = 0
        self.reward_sum = 0

    def step(self, action):
        """
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        action = int(action)
        reward = -1e-5

        if self.actions[action] == 'N':
            if self.row > 0:
                self.row -= 1
        elif self.actions[action] == 'E':
            if self.col < self.max_col - 1:
                self.col += 1
        elif self.actions[action] == 'S':
            if self.row < self.max_row - 1:
                self.row += 1
        elif self.actions[action] == 'V':
            if self.col > 0:
                self.col -= 1
        elif self.actions[action] == '+':
            self.state[self.row, self.col] += 0.1
        elif self.actions[action] == '-':
            self.state[self.row, self.col] -= 0.1
        elif self.actions[action] == 'R':
            self.state[self.row, self.col] += 0.1
        elif self.actions[action] == 'Q':
            old_max = self.observation.max()
            self.observation = self.query()
            if old_max == 0:
                reward += 1 - self.observation.max()
            else:
                reward += old_max - self.observation.max()
            self.q_idx += 1
            reward -= 0.01

        if self.observation.sum() > 0 and self.observation.argmax() != self.target_class:
            done = True
            reward += 10
        else:
            done = False

        if (self.state != 0).sum() >= 50:
            reward -= 1000
            done = True

        if done:
            self.episodes += 1
            self.reward_sum += reward
            if reward > 5:
                self.successes += 1
                self.n_queries += self.q_idx
            self.q_idx = 0

        return np.concatenate((self.observation, np.array([self.row / self.max_row,
                                                           self.col / self.max_col,
                                                (self.state != 0).sum() / 50]))).astype(np.float32), reward, done, {}

    def query(self):
        with torch.no_grad():
            return torch.softmax(self.net((self.image + self.state).cuda().clamp(-1, 1)), dim=1)[0].cpu().numpy()

    def reset(self):
        self.state *= 0
        self.observation = self.observation * 0
        self.row = self.max_row // 2
        self.col = self.max_col // 2
        start_obs = np.zeros(13, dtype=np.float32)
        start_obs[10] = self.row / self.max_row
        start_obs[11] = self.col / self.max_col
        return start_obs

    def reset_logger(self):
        self.episodes = 0
        self.successes = 0
        self.n_queries = 0
        self.q_idx = 0
        self.reward_sum = 0

    def set_data(self, image, label):
        self.image = image.unsqueeze(0)
        self.target_class = label

    def render(self, mode='human'):
        return

    def close(self):
        pass

    def seed(self, seed=None):
        return
