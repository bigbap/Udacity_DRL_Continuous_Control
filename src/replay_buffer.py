import numpy as np
from numpy import random as rnd
import torch

DEFAULT_BUFFER_SIZE = 1000
DEFAULT_BATCH_SIZE = 64


class Experience:
    def __init__(self, state, action, reward, state_prime, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.state_prime = state_prime
        self.done = done


class ReplayBuffer():
    def __init__(self, max_length=DEFAULT_BUFFER_SIZE, batch_size=DEFAULT_BATCH_SIZE, device="cpu"):
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.buffer = []
        self.head = -1

    def move_head(self):
        self.head = (self.head + 1) % self.max_length

        if len(self.buffer) < self.max_length:
            self.buffer += [None]

    def add(self, state, action, reward, state_prime, done):
        self.move_head()

        self.buffer[self.head] = Experience(
            state, action, reward, state_prime, done)

    def sample(self):
        batch = rnd.choice(self.buffer, size=self.batch_size)

        return self.prepare_batch(batch)

    def prepare_batch(self, batch):
        states = torch.from_numpy(
            np.vstack([e.state for e in batch if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in batch if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in batch if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.state_prime for e in batch if e is not None])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in batch if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)
