import itertools
import itertools

import numpy as np

from src.agent import DQNAgent
from src.environment import DQNEnvironment
from src.utils import preprocessing

from collections import deque, namedtuple

transition = namedtuple("Transition", field_names=("observation", "action", "reward", "next_observation", "done"))


def deep_q_learning(environment, config):

    # initialize environment and agent
    environment = DQNEnvironment(environment=environment, config=config)
    agent = DQNAgent(action_space=environment.action_space, config=config)

    episode_reward_buffer = deque(maxlen=100)

    n_steps = 1
    for episode in itertools.count():

        environment, agent, n_steps, episode_reward = deep_q_learning_episode(
            environment=environment,
            agent=agent,
            n_steps=n_steps,
            config=config)

        episode_reward_buffer.append(episode_reward)

        print(f'Episode: {episode}, Step: {n_steps}, Reward: {episode_reward}, Avg. Reward: {np.mean(episode_reward_buffer):.2f}')


def deep_q_learning_episode(environment, agent, n_steps, config):

    # initialize episode parameters
    episode_done = False
    episode_reward = 0.0
    n_episode_steps = 1

    # initialize environment and agent
    observation = environment.reset()
    agent.clear_observations()

    while not episode_done:

        # select action using a uniform random policy to populate the replay memory D; otherwise epsilon-greedy
        if not agent.replay_memory_is_full():
            action = agent.action_space.sample()
        else:
            action = agent.select_action(n_steps=n_steps)
            n_steps = n_steps + 1

        # execute action a_t in emulator
        observation, step_reward, episode_done, _ = environment.step(action)

        # observe image x_(t+1); store transition in replay memory D
        agent.append_observation(action, observation, step_reward, episode_done)

        # update the action-value function Q every k-th action
        if n_steps % config.update_frequency == 0:
            agent.update_network()

        # update target memory every k-th parameter upgrade
        if n_steps % (config.update_frequency * config.target_network_update_frequency) == 0:
            agent.update_target_network()

        episode_reward += step_reward
        n_episode_steps += 1

    return environment, agent, n_steps, episode_reward


