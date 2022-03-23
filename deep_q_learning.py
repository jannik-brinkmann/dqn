import argparse

from src.utils import create_dirs
import itertools

import numpy as np

from src.agent import DQNAgent
from src.environment import DQNEnvironment
from src.utils import set_seeds

from collections import deque


def deep_q_learning(environment, config):

    # initialize environment and agent
    environment = DQNEnvironment(environment=environment, config=config)
    agent = DQNAgent(action_space=environment.action_space, config=config)

    set_seeds(seed=config.seed)

    episode_reward_buffer = deque(maxlen=100)

    n_steps = 1
    for episode in itertools.count():

        # initialize episode parameters
        episode_done = False
        episode_reward = 0.0
        n_episode_steps = 1

        # initialize environment and agent
        observation = environment.reset(seed=config.seed)
        agent.clear_observations()

        while not episode_done:

            # select action using a uniform random policy to populate the replay memory D
            if not agent.replay_memory_is_full():
                action = agent.action_space.sample()
            # otherwise, use an epsilon-greedy strategy
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

        episode_reward_buffer.append(episode_reward)

        print(f'Episode: {episode}, Step: {n_steps}, Reward: {episode_reward}, Avg. Reward: {np.mean(episode_reward_buffer):.2f}')


if __name__ == '__main__':

    # see Extended Data Table 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--mini_batch_size', default=32)
    parser.add_argument('--replay_memory_size', default=50000)  # 1000000
    parser.add_argument('--agent_history_length', default=4)
    parser.add_argument('--target_network_update_frequency', default=10000)
    parser.add_argument('--gamma', default=0.99)  # discount factor
    parser.add_argument('--action_repeat', default=4)
    parser.add_argument('--update_frequency', default=4)
    parser.add_argument('--learning_rate', default=0.00025)
    parser.add_argument('--gradient_momentum', default=0.95)
    parser.add_argument('--squared_gradient_momentum', default=0.95)
    parser.add_argument('--min_squared_gradient', default=0.01)
    parser.add_argument('--epsilon_start', default=1)  # initial_epsilon
    parser.add_argument('--epsilon_end', default=0.1)  # final_epsilon
    parser.add_argument('--epsilon_decay', default=1000000)  # final_epsilon_frame
    parser.add_argument('--replay_start_size', default=25000)
    parser.add_argument('--no_op_max', default=30)

    # additional arguments
    parser.add_argument('--seed', default=42)
    config = parser.parse_args()

    # setup experiment
    create_dirs()

    try:
        # NoFrameskip - ensures no frames are skipped by the emulator
        # v4 - ensures actions are executed, whereas v0 would ignore the last action with 0.25 probability
        deep_q_learning(environment='BreakoutNoFrameskip-v4', config=config)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
