import numpy as np

seed = 123
np.random.seed(seed)
import random
import torch

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import torch.nn as nn
import torch.nn.functional as F


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), None, None

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.input_shape = input_shape  # softmax output
        self.num_actions = num_actions  # 2 ask for a label or not
        self.fc1 = nn.Linear(input_shape, self.num_actions * 4)
        self.fc2 = nn.Linear(self.num_actions * 4, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
