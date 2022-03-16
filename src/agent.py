from src.replay_memory import ReplayMemory
from src.model import DQNModel

import torch
import numpy as np
import random


class DQNAgent:

    def __init__(self, action_space, config):

        self.action_space = action_space
        self.config = config

        # initialize replay memory D to capacity N
        self.replay_memory = ReplayMemory(capacity=config.replay_memory_size)

        # initialize action-value function Q with random weights
        self.model = DQNModel(n_actions=action_space.n)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # initialize separate network for generating targets y_j as a clone of the action-value function Q
        self.target_model = DQNModel(n_actions=action_space.n)
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, observation, n_frame):
        epsilon = self._compute_epsilon(n_frame=n_frame)
        action = self._sample_action(observation=observation, epsilon=epsilon)
        return action, epsilon

    def _compute_epsilon(self, n_frame):
        return np.interp(n_frame, [0, self.config.epsilon_end_frame], [self.config.epsilon_start, self.config.epsilon_end])

    def _sample_action(self, observation, epsilon):
        choose_random = random.random() <= epsilon
        if choose_random:
            action = self.action_space.sample()
        else:
            action = self.model.act(observation)
        return action

    def sample_memories(self, size):

        # sample a random mini-batch of memories from replay memory D
        memories = self.replay_memory.sample(size=size)

        # unpack memories
        obses_t = torch.as_tensor(np.asarray([t.state for t in memories]), dtype=torch.float32)
        actions_t = torch.as_tensor(np.asarray([t.action for t in memories]), dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(np.asarray([t.reward for t in memories]), dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(np.asarray([t.done for t in memories]), dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(np.asarray([t.next_state for t in memories]), dtype=torch.float32)

        return obses_t, actions_t, rews_t, dones_t, new_obses_t

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
