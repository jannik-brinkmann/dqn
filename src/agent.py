from src.memory import ReplayMemory
from src.model import DQNModel

import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import itertools
import os


class DQNAgent:

    def __init__(self, action_space, config):
        self.action_space = action_space
        self.config = config

        # initialize observation buffer to store previous k observations
        self.observation_buffer = deque(maxlen=10)

        # initialize replay memory D with capacity N
        self.memory = ReplayMemory(capacity=config.replay_memory_size)

        # initialize action-value function Q with random weights
        self.model = DQNModel(n_actions=action_space.n)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=config.learning_rate)

        # initialize separate target network as a clone of the action-value function Q
        self.target_model = DQNModel(n_actions=action_space.n)
        self.target_model.load_state_dict(self.model.state_dict())

    def observe(self, observation):

        # store observation in observation buffer
        self.observation_buffer.append(observation)

    def store_experience(self, action, step_reward, episode_done):

        # append experience to replay memory if sufficient observations have been stored
        if len(self.observation_buffer) > self.config.agent_history_length:
            index = len(self.observation_buffer) - self.config.agent_history_length
            state = list(itertools.islice(self.observation_buffer, index - 1, index - 1 + 4))
            next_state = list(itertools.islice(self.observation_buffer, index, index + 4))

            step_reward = np.clip(step_reward, -1, 1)  # clip reward
            self.memory.append(state, action, step_reward, next_state, episode_done)

    def clear_observation_buffer(self):
        self.observation_buffer.clear()

    def select_action(self, mode):
        # in case the agent does not have enough observations to select an action ...
        if len(self.observation_buffer) < self.config.agent_history_length:

            # ... in training mode, select a random action a_t
            if mode == 'training':
                action = self.action_space.sample()

            # ... in inference mode, select 'do nothing' action
            else:
                action = 0

        # otherwise, select a_t = argmax(Q(phi(s_t)))
        else:
            index = len(self.observation_buffer) - self.config.agent_history_length
            state = np.array(list(itertools.islice(self.observation_buffer, index, index + 4)))
            q_values = self.model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            action = torch.argmax(q_values, dim=1)[0].detach().item()
        return action

    def sample_action(self, step, mode):

        # in training mode, with probability epsilon select a random action a_t
        epsilon = self._compute_epsilon(n_step=step)
        if mode == 'training' and random.random() < epsilon:
            action = self.action_space.sample()

        # in case the agent does not have enough observations to select an action ...
        else:
            action = self.select_action(mode=mode)
        return action

    def _compute_epsilon(self, n_step):
        return np.interp(n_step, (0, self.config.epsilon_decay), (self.config.epsilon_start, self.config.epsilon_end))

    def update_network(self):

        # sample a random mini-batch of memories from replay memory D
        states, actions, rewards, next_states, episodes_done = self._sample_memories(size=self.config.mini_batch_size)

        # compute targets y_j = r_j if episodes terminates at step j+1, otherwise y_j = r_j + gamma * max(Q_(j+1))
        target_q_values = self.target_model(next_states)
        targets = rewards + self.config.gamma * (1 - episodes_done) * target_q_values.max(dim=1, keepdim=True)[0]  # TODO: Replace with amax

        # compute loss (y_j - Q_j)^2
        q_values = torch.gather(input=self.model(states), dim=1, index=actions)
        loss = nn.functional.smooth_l1_loss(q_values, targets)  # TODO: Replace with Huber Loss

        # gradient descent step on computed loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _sample_memories(self, size):

        # sample a random mini-batch of memories from replay memory D
        memories = self.memory.sample(size=size)

        # unpack memories
        states = torch.as_tensor(np.asarray([t.state for t in memories]), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray([t.action for t in memories]), dtype=torch.int64).unsqueeze(-1)
        rewards = torch.as_tensor(np.asarray([t.reward for t in memories]), dtype=torch.float32).unsqueeze(-1)
        next_states = torch.as_tensor(np.asarray([t.next_state for t in memories]), dtype=torch.float32)
        episodes_done = torch.as_tensor(np.asarray([t.done for t in memories]), dtype=torch.float32).unsqueeze(-1)

        return states, actions, rewards, next_states, episodes_done

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model_weights(self, filename):
        os.makedirs(os.path.join(os.path.dirname(__file__), os.pardir, 'weights'), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(os.path.dirname(__file__), os.pardir, 'weights', filename))
        print("Model saved.")

    def load_model_weights(self, filename):
        try:
            self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), os.pardir, 'weights', filename)))
        except:
            print("Model cannot be loaded.")
