import argparse
import logging
import os

from src.deep_q_learning import deep_q_learning

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
    parser.add_argument('--weight_save_frequency', default=100)
    args = parser.parse_args()

    # setup directory to store weights
    os.makedirs(os.path.join(os.path.dirname(__file__), os.pardir, 'weights'), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - Episode %(episode)s Reward%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        # NoFrameskip - ensures no frames are skipped by the emulator
        # v4 - ensures actions are executed, whereas v0 would ignore an action with 0.25 probability
        deep_q_learning('BreakoutNoFrameskip-v4', args)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
