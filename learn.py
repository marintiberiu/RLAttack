import pfrl
import torch

from pfrl import replay_buffers, explorers, experiments, q_functions
from pfrl.agents import DQN
from pfrl.q_functions import DiscreteActionValueHead

from torch import nn, optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm

from rl.env_d import AttackEnv


if __name__ == '__main__':
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize(0.5, 0.5, 0.5)))
    test_dataset = CIFAR10('data', train=False, transform=transform, download=False)

    image_size = 32 * 32
    n_classes = 10

    env = AttackEnv()

    obs_space = env.observation_space
    action_space = env.action_space

    obs_size = obs_space.low.size

    n_actions = action_space.n

    q_func = pfrl.nn.RecurrentSequential(
        nn.Linear(obs_size, 64),
        nn.ReLU(),
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.LSTM(input_size=1024,
                hidden_size=512),
        nn.Linear(512, n_actions),
        DiscreteActionValueHead(),
    )
    # Use epsilon-greedy for exploration
    explorer = explorers.LinearDecayEpsilonGreedy(
        1,
        0.1,
        10 ** 4,
        action_space.sample,
    )

    opt = optim.Adam(q_func.parameters())
    rbuf = replay_buffers.EpisodicReplayBuffer(10 ** 6)

    agent = DQN(
        q_func,
        opt,
        rbuf,
        gpu=-1,
        gamma=0.99,
        explorer=explorer,
        recurrent=True,
        episodic_update_len=100
    )

    for data in tqdm(test_dataset):
        env.set_data(data[0], data[1])
        env.reset_logger()

        experiments.train_agent(
            agent=agent,
            env=env,
            steps=10000,
            outdir='output',
            max_episode_len=1000
        )

        print("Accuracy:", env.successes / env.episodes * 100,
              "Average reward:", env.reward_sum / env.episodes,
              "Average queries:", -1 if env.successes == 0 else env.n_queries / env.successes
              )

        torch.save(q_func.state_dict(), 'saves/q_model.sv')
