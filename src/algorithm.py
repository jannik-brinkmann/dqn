from torch import nn
import torch
from collections import deque
import itertools
import numpy as np
import random

GAMMA = 0.99
BATCH_SIZE = 32


from src.environment import DQNEnvironment
from src.agent import DQNAgent


def deep_q_learning(config):

    # initialize environment
    environment = DQNEnvironment('CartPole-v0')
    observation = environment.reset()

    # initialize agent
    agent = DQNAgent(action_space=environment.action_space, config=config)

    # uniform random policy to populate the replay memory D
    for _ in range(config.replay_start_size):
        action = agent.action_space.sample()
        next_observation, reward, done, _ = environment.step(action=action)
        agent.replay_memory.append(observation, action, reward, next_observation, done)
        observation = environment.reset() if done else next_observation

    n_frames = 0

    for episode in itertools.reset():
        
        episode_reward = 0.0

    # main train loop
    observation = environment.reset()
    for step in itertools.count():
        action, epsilon = agent.select_action(observation=observation, n_frame=step)


        next_observation, reward, done, _ = environment.step(action)
        agent.replay_memory.append(observation, action, reward, next_observation, done)
        observation = next_observation

        episode_reward += reward

        if done:
            observation = environment.reset()
            agent.episode_reward_buffer.append(episode_reward)
            episode_reward = 0.0

        # update the action-value function Q every k-th action
        #if step % config.update_frequency == 0:
        agent.learn()

        # upd target net
        if step % (config.target_network_update_frequency) == 0:
            agent.update_target_network()

        if step % 1000 == 0:
            print()
            print('Step', step)
            print('Avg. Reward', np.mean(agent.episode_reward_buffer))

