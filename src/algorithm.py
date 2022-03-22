import itertools
import itertools

import numpy as np

from src.agent import DQNAgent
from src.environment import DQNEnvironment
from src.utils import preprocessing

from collections import deque, namedtuple

transition = namedtuple("Transition", field_names=("observation", "action", "next_observation"))


def deep_q_learning(environment, config):

    # initialize environment and agent
    environment = DQNEnvironment(environment=environment)
    agent = DQNAgent(action_space=environment.action_space, config=config)

    observation = environment.reset()

    # uniform random policy to populate the replay memory D
    while not agent.replay_memory_is_full():
        action = agent.action_space.sample()
        next_observation, step_reward, episode_done, _ = environment.step(action=action)
        agent.observe_transition(observation=observation, action=action, next_observation=next_observation)
        agent.store_experience(action, step_reward, episode_done)
        observation = next_observation
        if episode_done:
            observation = environment.reset()
            agent.reset(observation=observation)

    episode_reward_buffer = deque(maxlen=100)

    n_steps = 1
    for episode in itertools.count():

        environment, agent, n_steps, episode_reward = deep_q_learning_episode(environment, agent, n_steps, config, episode, episode_reward_buffer)

        episode_reward_buffer.append(episode_reward)


def deep_q_learning_episode(environment, agent, n_steps, config, episode, episode_reward_buffer):

    # initialize episode parameters
    episode_done = False
    episode_reward = 0.0
    n_episode_steps = 1

    # initial observation
    observation = environment.reset()
    agent.reset(observation=observation)

    while not episode_done:

        # select action every k-th step (frame-skipping technique)
        if n_episode_steps % config.action_repeat == 0:
            if not agent.replay_memory_is_full():
                action = agent.action_space.sample()  # uniform random policy to populate the replay memory D
            else:
                action = agent.select_action(n_steps=n_steps)  # epsilon-greedy strategy
                n_steps = n_steps + 1

            # execute action a_t in emulator
            next_observation, step_reward, episode_done, _ = environment.step(action)

            # observe transition x_t, a_t, x_(t+1)
            agent.observe_transition(transition(observation, action, next_observation))

            # store transition in replay memory D
            agent.store_experience(action, step_reward, episode_done)

        # repeat action on skipped frames (frame-skipping technique)
        else:
            observation, step_reward, _, _ = environment.step(action)

        # update the action-value function Q every k-th action
        if n_steps % (config.action_repeat * config.update_frequency) == 0:
            agent.update_network()

        # update target memory every k-th parameter upgrade
        if n_steps % (config.action_repeat * config.update_frequency * config.target_network_update_frequency) == 0:
            agent.update_target_network()

        if n_steps % 1000 == 0:
            print(f'Episode: {episode}, Step: {n_steps}, Avg. Reward: {np.mean(episode_reward_buffer):.2f}')

        episode_reward += step_reward

    return environment, agent, n_steps, episode_reward


