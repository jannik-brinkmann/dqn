import argparse

import gym

from dqn.algorithm import deep_q_learning


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
    parser.add_argument('--replay_start_size', default=50000)  # 50000
    parser.add_argument('--max_n_wait_actions', default=30)  # no_op_max

    # see caption of Extended Data Table 3
    parser.add_argument('--n_training_steps', default=10000000)
    parser.add_argument('--evaluation_frequency', default=250000)
    parser.add_argument('--n_evaluation_steps', default=135000)
    args = parser.parse_args()

    # NoFrameskip - ensures no frames are skipped by the emulator
    # v4 - ensures actions are executed, whereas v0 would ignore an action with 0.25 probability
    environments = ['BreakoutNoFrameskip-v4']

    for environment in environments:

        training_score = deep_q_learning(environment=environment, args=args)

        print(f'{environment} Score: {training_score}')
