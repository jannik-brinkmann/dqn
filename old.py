def _add_observation(self, observation):
    if self.observation_buffer:
        self.observation_buffer.append(observation)
    else:
        self.observation_buffer.extend(np.full((self.observation_buffer.maxlen, 4), observation))

def add_observation(self, observation):

    # add observation to sequence s_t and preprocessed sequence phi_t
    if self.observation_buffer and self.preprocessed_sequence:
        self.observation_buffer.append(observation)
        self.preprocessed_sequence.append(preprocess_sequence(observation, observation))

    # in the initial step of each episode, to populate sequence s_t and preprocessed sequence phi_t
    else:
        self.observation_buffer.extend(np.full((self.observation_buffer.maxlen, 210, 160, 3), observation))
        self.preprocessed_sequence.extend(np.full((self.observation_buffer.maxlen, 84, 84), preprocess_sequence(self.observation_buffer[-2], self.observation_buffer[-1])))

def get_state(self):
    return self.preprocessed_sequence[-self.config.agent_history_length:]


# initialize sequence s_t and preprocessed sequence phi_t
self.observation_buffer = deque(maxlen=10)
self.preprocessed_sequence = deque(maxlen=10)
