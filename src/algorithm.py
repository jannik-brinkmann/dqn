from torch import nn
import torch
from collections import deque
import itertools
import numpy as np
import random

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000


from src.model import DQNModel
from src.replay_memory import ReplayMemory
from src.environment import DQNEnvironment
from src.agent import DQNAgent


def deep_q_learning(config):

    # initialize environment
    environment = DQNEnvironment('CartPole-v0')
    observation = environment.reset()

    # initialize agent
    agent = DQNAgent(action_space=environment.action_space, config=config)

    rew_buffer = deque([0.0], maxlen=100)

    n_frames = 0

    episode_reward = 0.0

    for _ in range(MIN_REPLAY_SIZE):
        action = environment.action_space.sample()

        new_obs, rew, done, _ = environment.step(action)
        agent.replay_memory.append(observation, action, rew, new_obs, done)
        observation = new_obs

        if done:
            observation = environment.reset()

    # main train loop
    observation = environment.reset()
    for step in itertools.count():
        action, epsilon = agent.select_action(observation=observation, n_frame=step)


        new_obs, rew, done, _ = environment.step(action)
        agent.replay_memory.append(observation, action, rew, new_obs, done)
        observation = new_obs

        episode_reward += rew

        if done:
            observation = environment.reset()
            rew_buffer.append(episode_reward)
            episode_reward = 0.0

        # start gradient step
        obses_t, actions_t, rews_t, dones_t, new_obses_t = agent.sample_memories(size=config.mini_batch_size)

        # compute targets
        target_q_values = agent.target_model(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + config.gamma * (1 - dones_t) * max_target_q_values

        # compute loss
        q_values = agent.model(obses_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        # grad des
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        # upd target net
        if step % config.target_network_update_frequency == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        if step % 1000 == 0:
            print()
            print('Step', step)
            print('Avg. Reward', np.mean(rew_buffer))

