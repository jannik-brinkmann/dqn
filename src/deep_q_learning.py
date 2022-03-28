import argparse
import itertools
import os
from collections import deque
import logging

import numpy as np

from src.agent import DQNAgent
from src.environment import DQNEnvironment

from torch.utils.tensorboard import SummaryWriter
import gym


def deep_q_learning(environment, config):

    writer = SummaryWriter('logs')

    if config.mode == 'training':
        environment = DQNEnvironment(environment=gym.make(environment), config=config)
    else:
        environment = DQNEnvironment(environment=gym.make(environment, render_mode='human'), config=config)

    agent = DQNAgent(action_space=environment.action_space, config=config)

    n_steps = 1  # no. of actions selected by the agent
    for episode in itertools.count():

        # start a new episode after loss-of-live or loss-of-game
        environment.reset(seed=config.seed)
        agent.clear_observations()

        # set episode parameters
        episode_done = False
        episode_reward = 0.0

        while not episode_done:

            # select action using a uniform random policy to populate the replay memory D
            if config.mode == 'training' and not agent.memory.is_full():
                action = agent.action_space.sample()

            # otherwise, select action using an epsilon-greedy strategy
            else:
                action = agent.select_action(n_steps=n_steps)
                n_steps = n_steps + 1

            # execute action a_t in emulator
            observation, step_reward, episode_done, info = environment.step(action)

            # observe image x_(t+1); store transition in replay memory Dy
            agent.append_observation(action, observation, step_reward, episode_done)

            # update the action-value function Q every k-th action
            if config.mode == 'training' and n_steps % config.update_frequency == 0:
                loss = agent.update_network()
                writer.add_scalar('Loss', loss, n_steps)

            # update target network every k-th update of the action-value function Q
            if config.mode == 'training' and n_steps % (config.update_frequency * config.target_update_frequency) == 0:
                agent.update_target_network()

            episode_reward += step_reward

        writer.add_scalar('Episode Reward', episode_reward, episode)

        print(f'Episode: {episode}, Step: {n_steps}, Reward: {episode_reward}')

        # store network weights every k-th episode
        if config.mode == 'training' and episode % config.weight_save_frequency == 0:
            agent.save_model_weights('checkpoint_' + str(episode) + '.pth.tar')