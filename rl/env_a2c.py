import os
from queue import LifoQueue

import gym
import numpy as np
import torch
from torchvision.models import resnet18
from PIL import Image
from random import randint


class AttackEnv(gym.core.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, 32, 32), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(32 * 32)
        self.image = None
        self.target_class = None
        self.net = resnet18(pretrained=False, num_classes=10).eval().cuda()
        self.net.load_state_dict(torch.load('saves/resnet18_cifar10.sv'))
        self.state = np.zeros((32, 32), dtype=np.float32)
        self.action_list = LifoQueue()
        # logging tools
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
        row = action // 32
        col = action % 32

        if self.action_list.qsize() == 50:
            f_row, f_col = self.action_list.get()
            self.state[f_row, f_col] = 0

        self.state[row, col] = 1
        self.action_list.put((row, col))

        target, best = self.query()
        reward = target - best

        self.q_idx += 1

        if best > target:
            done = True
            reward = 10
        else:
            done = False

        if done:
            self.reward_sum += reward
            if reward > 5:
                self.successes += 1
                self.n_queries += self.q_idx
            self.q_idx = 0

        return (self.image + self.state)[0], reward, done, {}

    def query(self):
        with torch.no_grad():
            q = torch.softmax(self.net((self.image + self.state).cuda().clamp(-1, 1)), dim=1)[0].cpu().numpy()
            target = q[self.target_class]
            q[self.target_class] = 0
            return target, q.max()

    def reset(self):
        self.state *= 0
        return self.image[0]

    def reset_logger(self):
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
