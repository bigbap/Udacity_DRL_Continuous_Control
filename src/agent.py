from src.model import Actor, Critic
from src.replay_buffer import ReplayBuffer
from src.noise import OUNoise
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# solved in 180 episodes
BATCH_SIZE = 64
REPLAY_LENGTH = 10000
LEARN_EVERY = 1
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 2e-4
LR_CRITIC = 2e-4
WEIGHT_DECAY = 0.0001

# BATCH_SIZE = 32
# REPLAY_LENGTH = 5000
# LEARN_EVERY = 1
# GAMMA = 0.99
# TAU = 1e-3
# LR_ACTOR = 2e-4
# LR_CRITIC = 2e-4
# WEIGHT_DECAY = 0.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, s_dim, a_dim, seed=0, model=None):
        # actor netowrks
        self.actor_local = Actor(
            s_dim=s_dim, a_dim=a_dim, seed=seed).to(device)
        self.actor_target = Actor(
            s_dim=s_dim, a_dim=a_dim, seed=seed).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=LR_ACTOR)
        self.clone_weights(self.actor_target, self.actor_local)

        # critic netowrks
        self.critic_local = Critic(
            s_dim=s_dim, a_dim=a_dim, seed=seed).to(device)
        self.critic_target = Critic(
            s_dim=s_dim, a_dim=a_dim, seed=seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.clone_weights(self.critic_target, self.critic_local)

        # replay buffer
        self.replay_buffer = ReplayBuffer(
            REPLAY_LENGTH, batch_size=BATCH_SIZE, device=device)

        # noise process
        self.noise = OUNoise(a_dim, seed)

        self.t_step = -1

    def clone_weights(self, w1, w0):
        for p1, p0 in zip(w1.parameters(), w0.parameters()):
            p1.data.copy_(p0.data)

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        state = torch.tensor(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def step(self, state, action, reward, state_prime, done):
        self.t_step += 1
        self.replay_buffer.add(state, action, reward, state_prime, done)

        if len(self.replay_buffer) >= BATCH_SIZE and self.t_step % LEARN_EVERY == 0:
            self.learn()
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

    def learn(self):
        states, actions, rewards, states_prime, dones = self.replay_buffer.sample()

        # update critic
        action_prime = self.actor_target(states_prime)
        prediction = self.critic_local(states, actions)
        target_prime = self.critic_target(states_prime, action_prime)
        target = rewards + (GAMMA * target_prime * (1 - dones))

        loss = F.mse_loss(prediction, target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # update actor
        actions_prediction = self.actor_local(states)

        loss = -self.critic_local(states, actions_prediction).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0-tau) * target_param.data)

    def save_weights(self):
        torch.save(self.actor_local.state_dict(), 'actor_local_model.pth')
        torch.save(self.critic_local.state_dict(), 'critic_local_model.pth')

    def load_weights(self):
        self.actor_local.load_state_dict(torch.load('actor_local_model.pth'))
        self.critic_local.load_state_dict(torch.load('critic_local_model.pth'))
