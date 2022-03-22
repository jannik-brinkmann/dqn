from src.replay_memory import ReplayMemory
from src.model import DQNModel, DQNModelCartPole

import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque, namedtuple
from src.utils import preprocessing
import itertools

import cv2

sequence_element = namedtuple("SequenceElement", field_names=("observation", "action", "next_observation"))


class DQNAgent:

    def __init__(self, action_space, config):

        self.action_space = action_space
        self.config = config

        self.observation_buffer = deque(maxlen=10)

        # initialize replay memory D with capacity N
        self.replay_memory = ReplayMemory(capacity=config.replay_memory_size)

        # initialize action-value function Q with random weights
        self.model = DQNModel(n_actions=action_space.n)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # initialize separate network for generating targets y_j as a clone of the action-value function Q
        self.target_model = DQNModel(n_actions=action_space.n)
        self.target_model.load_state_dict(self.model.state_dict())

    def append_observation(self, action, observation, step_reward, episode_done):
        self.observation_buffer.append(observation)

        if len(self.observation_buffer) > self.config.agent_history_length:
            index = len(self.observation_buffer) - self.config.agent_history_length
            state = list(itertools.islice(self.observation_buffer, index - 1, index - 1 + 4))
            next_state = list(itertools.islice(self.observation_buffer, index, index + 4))

            self.replay_memory.append(state, action, step_reward, next_state, episode_done)

    def clear_observations(self):
        self.observation_buffer.clear()

    def observe_transition(self, transition):
        self.observation_buffer.append(preprocessing(transition.observation, transition.next_observation))

        if len(self.observation_buffer) > self.config.agent_history_length:

            index = len(self.observation_buffer) - self.config.agent_history_length
            state = list(itertools.islice(self.observation_buffer, index - 1, index - 1 + 4))
            next_state = list(itertools.islice(self.observation_buffer, index, index + 4))

            self.replay_memory.append(state, transition.action, transition.reward, next_state, transition.done)

    def replay_memory_is_full(self):
        return self.replay_memory.is_full()

    def select_action(self, n_steps):
        # with probability epsilon select a random action a_t
        epsilon = self._compute_epsilon(n_step=n_steps)
        if random.random() < epsilon:
            action = self.action_space.sample()
        # otherwise, select a_t = argmax(Q(phi(s_t)))
        else:
            index = len(self.observation_buffer) - self.config.agent_history_length
            state = np.array(list(itertools.islice(self.observation_buffer, index, index + 4)))
            q_values = self.model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            action = torch.argmax(q_values, dim=1)[0].detach().item()
        return action

    def _compute_epsilon(self, n_step):
        return np.interp(n_step, (0, self.config.epsilon_decay), (self.config.epsilon_start, self.config.epsilon_end))

    def update_network(self):

        # sample a random mini-batch of memories from replay memory D
        states, actions, rewards, next_states, episodes_done = self._sample_memories(size=self.config.mini_batch_size)

        # compute targets
        target_q_values = self.target_model(next_states)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rewards + self.config.gamma * (1 - episodes_done) * max_target_q_values

        # compute loss
        q_values = self.model(states)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        # grad des
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _sample_memories(self, size):

        # sample a random mini-batch of memories from replay memory D
        memories = self.replay_memory.sample(size=size)

        # unpack memories
        states = torch.as_tensor(np.asarray([t.state for t in memories]), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray([t.action for t in memories]), dtype=torch.int64).unsqueeze(-1)
        rewards = torch.as_tensor(np.asarray([t.reward for t in memories]), dtype=torch.float32).unsqueeze(-1)
        next_states = torch.as_tensor(np.asarray([t.next_state for t in memories]), dtype=torch.float32)
        episodes_done = torch.as_tensor(np.asarray([t.done for t in memories]), dtype=torch.float32).unsqueeze(-1)

        return states, actions, rewards, next_states, episodes_done

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
