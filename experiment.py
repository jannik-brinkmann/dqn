import argparse
import logging
import os

from src.algorithm import deep_q_learning
from torch.utils.tensorboard import SummaryWriter
import warnings
import gym

from src.agent import DQNAgent
from src.environment import DQNEnvironment


def experiment(environment, mode):

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    # see Extended Data Table 1
    parser.add_argument('--mini_batch_size', default=32)
    parser.add_argument('--replay_memory_size', default=100000)  # 1000000
    parser.add_argument('--agent_history_length', default=4)
    parser.add_argument('--target_update_frequency', default=10000)  # target_network_update_frequency
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
    parser.add_argument('--replay_start_size', default=25000)  # 50000
    parser.add_argument('--max_n_wait_actions', default=30)  # no_op_max

    # additional arguments
    parser.add_argument('--mode', default='training')
    config = parser.parse_args()

    # in training mode, hide display to increase speed of simulation
    render_mode = 'rgb_array' if config.mode == 'training' else 'human'
    environment = DQNEnvironment(environment=gym.make(environment, render_mode=render_mode), config=config)
    agent = DQNAgent(action_space=environment.action_space, config=config)

    deep_q_learning(agent, environment, config)


if __name__ == '__main__':
    # NoFrameskip - ensures no frames are skipped by the emulator
    # v4 - ensures actions are executed, whereas v0 would ignore an action with 0.25 probability
    experiment(environment='BreakoutNoFrameskip-v4')
