import itertools
import itertools

import numpy as np

from src.agent import DQNAgent
from src.environment import DQNEnvironment


def deep_q_learning(config):

    # initialize environment
    environment = DQNEnvironment('CartPole-v0')
    observation = environment.reset()

    # initialize agent
    agent = DQNAgent(action_space=environment.action_space, config=config)

    # uniform random policy to populate the replay memory D
    for _ in range(config.replay_start_size):
        action = agent.action_space.sample()
        next_observation, reward, episode_done, _ = environment.step(action=action)
        agent.replay_memory.append(observation, action, reward, next_observation, episode_done)
        observation = environment.reset() if episode_done else next_observation

    n_step = 1
    for episode in itertools.count():

        episode_done = False
        episode_reward = 0.0
        observation = environment.reset()

        while not episode_done:

            # select action every k-th frame and repeat action on skipped frames (frame-skipping technique)
            action, epsilon = agent.sample_action(observation=observation, n_frame=n_step)

            # execute action a_t in emulator; observe reward r_t and image x_(t+1)
            next_observation, reward, episode_done, _ = environment.step(action)
            episode_reward += reward

            # store memories every k-th frame (frame-skipping technique)
            agent.replay_memory.append(observation, action, reward, next_observation, episode_done)
            observation = next_observation

            # update the action-value function Q every k-th action
            #if step % config.update_frequency == 0:
            agent.update_network()

            # update target memory every k-th parameter upgrade
            if n_step % config.target_network_update_frequency == 0:
                agent.update_target_network()

            if n_step % 1000 == 0:
                print(f'Episode: {episode}, Step: {n_step}, Avg. Reward: {np.mean(agent.episode_reward_buffer):.2f}')

            n_step = n_step + 1

        agent.episode_reward_buffer.append(episode_reward)
