import argparse
import random

from src.algorithm import deep_q_learning


def experiment(environment):

    random.seed(42)

    # see Extended Data Table 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--mini_batch_size', default=32)
    parser.add_argument('--replay_memory_size', default=10000)  # 1000000
    parser.add_argument('--agent_history_length', default=4)
    parser.add_argument('--target_network_update_frequency', default=10000)  # 10000
    parser.add_argument('--gamma', default=0.99)  # discount factor
    parser.add_argument('--action_repeat', default=4)
    parser.add_argument('--update_frequency', default=4)
    parser.add_argument('--learning_rate', default=0.00025)
    parser.add_argument('--gradient_momentum', default=0.95)
    parser.add_argument('--squared_gradient_momentum', default=0.95)
    parser.add_argument('--min_squared_gradient', default=0.01)
    parser.add_argument('--epsilon_start', default=1)  # initial_epsilon
    parser.add_argument('--epsilon_end', default=0.1)  # final_epsilon
    parser.add_argument('--epsilon_decay', default=100000)  # final_epsilon_frame, 1000000
    parser.add_argument('--replay_start_size', default=5000)
    parser.add_argument('--no_op_max', default=30)
    config = parser.parse_args()

    try:
        deep_q_learning(environment=environment, config=config)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')


if __name__ == '__main__':
    experiment(environment='BreakoutNoFrameskip-v4')
