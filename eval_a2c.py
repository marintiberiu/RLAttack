import csv

import pfrl
import torch

from pfrl import replay_buffers, explorers, experiments, q_functions
from pfrl.agents import DQN, a2c
from pfrl.q_functions import DiscreteActionValueHead

from torch import nn, optim
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm, trange

from a2c_net import A2CNet
from rl.env_a2c import AttackEnv

from PIL import Image


if __name__ == '__main__':
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize(0.5, 0.5, 0.5)))
    test_dataset = CIFAR10('data', train=False, transform=transform, download=False)
    test_dataset = Subset(test_dataset, range(9000, 9200))

    image_size = 32 * 32
    n_classes = 10
    max_episodes = 10
    max_episode_len = 100

    f = open("results/a2c_v2_100_n.csv", 'w', newline='')
    csv_w = csv.writer(f)
    csv_w.writerow(("N Correct", "Queries"))

    env = AttackEnv()

    obs_space = env.observation_space
    action_space = env.action_space

    obs_size = obs_space.low.size

    n_actions = action_space.n

    model = A2CNet()
    model.load_state_dict(torch.load('saves/best_a2c_model.sv'))

    optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
        model.parameters(),
        lr=1e-5,
        eps=1e-5,
        alpha=0.99,
    )
    rbuf = replay_buffers.EpisodicReplayBuffer(10 ** 6)

    agent = a2c.A2C(
        model,
        optimizer,
        gamma=0.99,
        gpu=0,
        num_processes=3,
        update_steps=5,
        phi=lambda x: np.asarray(x, dtype=np.float32) / 255,
    )
    agent.training = False

    for idx, data in enumerate(test_dataset):
        tqdm.write(" Image " + str(idx))
        env.set_data(data[0], data[1])

        env.reset_logger()
        total_queries = 0
        for ep in trange(max_episodes):
            obs = env.reset()
            env.n_queries = 0
            while env.n_queries < max_episode_len:
                action = agent.act(obs)
                obs, r, done, info = env.step(action)
                agent.observe(obs, r, done, env.n_queries == max_episode_len - 1)
                if done:
                    img = env.image[0]
                    img_2 = env.image[0] + env.state[0]
                    img = (img + 1) * 0.5 * 255
                    img_2 = (img_2 + 1) * 0.5 * 255
                    img = img.permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
                    img_2 = img_2.permute((1, 2, 0)).cpu().numpy().astype(np.uint8)

                    q = env.net((env.image + env.state).cuda().clamp(-1, 1))[0]
                    Image.fromarray(img).save("D:\\images\\img_" + str(idx) + '_' + str(env.target_class) + ".png")
                    Image.fromarray(img_2).save("D:\\images\\img_" + str(idx) + '_' + str(q.argmax().item()) + "_2.png")
                    break
            agent.observe(obs, -1, True, False)
            total_queries += env.n_queries

        csv_w.writerow((env.successes,  total_queries))
        f.flush()