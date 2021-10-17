import torch.nn as nn
import torch


class Agent(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Agent, self).__init__()

        layers = [
            nn.Linear(s_dim, 64),
            nn.ReLU(),
            nn.Linear(64, a_dim),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def act(self, state, ep=None):
        state = torch.tensor(state, dtype=torch.float)

        actions = self.forward(state)

        return actions

    def step(self, *args, **kwargs):
        pass
