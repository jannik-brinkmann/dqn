import random
from collections import deque, namedtuple

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def append(self, state, action, reward, next_state, done):
        self.memory.append(experience(state, action, reward, next_state, done))

    def sample(self, size):
        return random.sample(self.memory, size)
