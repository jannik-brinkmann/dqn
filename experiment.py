import argparse
import logging
import os

from src.algorithm import deep_q_learning
from torch.utils.tensorboard import SummaryWriter
import warnings
import gym

from src.agent import DQNAgent
from src.environment import DQNEnvironment
from datetime import datetime


if __name__ == '__main__':

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

    # see Caption of Extended Data Table 3
    parser.add_argument('--n_training_steps', default=10000000)
    parser.add_argument('--evaluation_frequency', default=250000)
    parser.add_argument('--n_evaluation_steps', default=135000)
    args = parser.parse_args()

    games = ['Breakout', 'Enduro', 'Riverraid', 'Seaquest', 'Spaceinvaders']

    for game in games:
        experiment_name = datetime.today().strftime('%Y-%m-%d') + '_' + game

        # NoFrameskip - ensures no frames are skipped by the emulator
        # v4 - ensures actions are executed, whereas v0 would ignore an action with 0.25 probability
        max_avg_episode_score = deep_q_learning(environment_name=game, experiment_name=experiment_name, args=args)

        print(f'{game} Score: {max_avg_episode_score}')
