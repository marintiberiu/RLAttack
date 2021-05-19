import pfrl
import torch

from pfrl import replay_buffers, explorers, experiments, q_functions
from pfrl.agents import DQN, a2c
from pfrl.q_functions import DiscreteActionValueHead

from torch import nn, optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm, trange

from a2c_net import A2CNet
from rl.env_a2c import AttackEnv


if __name__ == '__main__':
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize(0.5, 0.5, 0.5)))
    test_dataset = CIFAR10('data', train=False, transform=transform, download=False)

    image_size = 32 * 32
    n_classes = 10
    max_episodes = 20
    max_episode_len = 1000

    env = AttackEnv()

    obs_space = env.observation_space
    action_space = env.action_space

    obs_size = obs_space.low.size

    n_actions = action_space.n

    model = A2CNet()

    # Use epsilon-greedy for exploration
    explorer = explorers.LinearDecayEpsilonGreedy(
        1,
        0.1,
        10 ** 4,
        action_space.sample,
    )

    optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
        model.parameters(),
        lr=1e-3,
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

    for idx, data in enumerate(test_dataset):
        tqdm.write(" Image " + str(idx))
        env.set_data(data[0], data[1])

        env.reset_logger()

        for ep in trange(max_episodes):
            obs = env.reset()
            for k in range(max_episode_len):
                action = agent.act(obs)
                obs, r, done, info = env.step(action)
                agent.observe(obs, r, done, k == max_episode_len - 1)
                if done:
                    break

        print("Accuracy:", env.successes / max_episodes * 100,
              "Average reward:", env.reward_sum / max_episodes,
              "Average queries:", -1 if env.successes == 0 else env.n_queries / env.successes
              )

        torch.save(model.state_dict(), 'saves/a2c_model.sv')
