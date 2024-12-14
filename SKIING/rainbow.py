import gymnasium as gym
import ale_py
import shimmy

import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, deque
from copy import deepcopy, copy

import cv2
import timeit

from argparse import ArgumentParser
import os
from PIL import Image
import pickle

import wandb


# Create the environment
env = gym.make("ALE/Skiing-v5", render_mode="rgb_array")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR = 0.001
MEMORY_SIZE = 8452*2
MAX_EPISODES = 5000
EPSILON = 0.99
EPSILON_DECAY = 0.999
GAMMA = 1 #0.99
BATCH_SIZE = 64
BURN_IN = 1000
DNN_UPD = 1
DNN_SYNC = 2500
CHANNELS = 1
WINDOW_SIZE = 80

NUM_ACTIONS = 3  # Number of actions the agent can take
INPUT_SHAPE = (CHANNELS, WINDOW_SIZE, WINDOW_SIZE)  # 4 x 80 x 80 # PyTorch uses (channels, height, width) format

HIDDEN_SIZE = 256
n_actions = env.action_space.n

# Preprocess function for images
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Normalize
    normalized = (resized / 255.0) * 2 - 1
    return normalized

class DDQN(torch.nn.Module):
    def __init__(self, env, learning_rate=1e-3, device='cpu'):
        super(DDQN, self).__init__()
        self.device = device
        self.n_outputs = NUM_ACTIONS
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        
        # Convolutional layers for processing image input
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the output from conv
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1 # fomula
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        
        linear_input_size = convw * convh * 64
        
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, n_actions)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
                
        if self.device == 'cuda':
            self.to(self.device)
                
    def forward(self, x):
        x = x.to(self.device)
        conv_out = self.conv(x)
        flattened = conv_out.view(x.size(0), -1)
        return self.fc(flattened)
    
    def get_qvals(self, state):
        # Convert state to proper shape and type
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Ensure correct shape (B, C, H, W)
        if state.dim() == 3:
            state = state.unsqueeze(1)  # Add batch dimension
        if state.dim() == 2:
            state = state.unsqueeze(0).unsqueeze(1)
            
        return self(state)
    
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            return np.random.choice(self.actions)
        else:
            with torch.no_grad():
                qvals = self.get_qvals(state)
                return torch.argmax(qvals).item()


# DDQN Agent class to control the training
class DDQNAgent:
    def __init__(self, env, dnnetwork, buffer, epsilon=0.1, eps_decay=0.99, batch_size=32):
        self.env = env
        self.dnnetwork = dnnetwork
        self.target_network = deepcopy(dnnetwork)  # Copy of the main network as target
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.nblock = 100  # Number of episodes for calculating average reward
        self.reward_threshold = self.env.spec.reward_threshold
        self.initialize()
    
    def initialize(self):
        self.update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.state0 = preprocess_frame(self.env.reset()[0])
        
        self.dnnetwork.to(self.dnnetwork.device)
        self.target_network.to(self.dnnetwork.device)
        
    # Take a step in the environment and store experience
    def take_step(self, eps, mode='train'):
        if mode == 'explore':
            action = self.env.action_space.sample()  # Random action during exploration
        else:
            action = self.dnnetwork.get_action(self.state0, eps)  # Q-value based action
            self.step_count += 1
            
        img = env.render()
            
        # Execute the action and get the new state and reward
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        #print(f"Reward: {reward}")
        new_state = preprocess_frame(new_state)
        done = terminated or truncated
        self.total_reward += reward
        
        # Store the experience in the replay buffer
        self.buffer.append(
            self.state0,
            action,
            reward,
            done,
            new_state
        )
        
        self.state0 = new_state  # Update the current state
        if done:
            self.state0 = preprocess_frame(self.env.reset()[0])
        return done, img
    
    # Training loop
    def train(self, gamma=0.99, max_episodes=50000, batch_size=32, dnn_update_frequency=4, dnn_sync_frequency=2000, max_timesteps=1000):
        self.gamma = gamma

        # Fill the buffer with random experiences
        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')
            
        episode = 0
        training = True
        
        best_score_mean = -np.inf # To check what model to save
        best_score_episode = -np.inf
        
        print("Training...")
        while training:
            
            images = []
            self.total_reward = 0
            gamedone = False
            frames = 0 # To check the length of the epsiode. 
            
            while not gamedone:
                
                frames += 1
                
                # Take a step
                gamedone, img = self.take_step(self.epsilon, mode='train')
                if img is not None:  # Check if the render method returns a valid image
                    images.append(Image.fromarray(img))
                
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                
                # Update the main network periodically
                if self.step_count % dnn_update_frequency == 0:
                    self.target_network.load_state_dict(self.dnnetwork.state_dict())
                    self.sync_eps.append(episode)
                                    
                if gamedone:
                    episode += 1
                    self.training_rewards.append(self.total_reward)

                    if len(self.training_rewards) >= self.nblock:
                        mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    elif len(self.training_rewards) > 0:
                        mean_rewards = np.mean(self.training_rewards)
                    else:
                        mean_rewards = 0.0  # Default mean when no rewards are present

                    self.mean_training_rewards.append(mean_rewards)
                    
                    if self.total_reward > best_score_episode: # Best episode score
                        best_score_episode = self.total_reward
                        if images:
                            # Ensure the folder exists
                            folder_path = "gifs_rainbow"
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)

                            # Save the GIF
                            images[0].save(f"{folder_path}/best_reward_{self.total_reward}.gif", save_all=True, append_images=images[1:], loop=0)
                    
                    if mean_rewards > best_score_mean: # Save the best model
                        best_score_mean = mean_rewards
                        torch.save(self.dnnetwork.state_dict(), "model_rainbow.pth")

                    print(f"\rEpisode {episode} Mean Rewards {mean_rewards:.2f} Epsilon {self.epsilon}", end="")
                    
                    if episode % 100 == 0:
                        if images:
                            # Ensure the folder exists
                            folder_path = "gifs_rainbow"
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)

                            # Save the GIF
                            images[0].save(f"{folder_path}/episode_{episode}_reward_{self.total_reward}.gif", save_all=True, append_images=images[1:], loop=0)

                    
                    # Log to wandb
                    wandb.log({
                        'episode': episode,
                        'mean_rewards': mean_rewards,
                        'episode reward': self.total_reward,
                        'epsilon': self.epsilon,
                        'loss': np.mean(self.update_loss), 
                        'frames': frames,
                        'best score mean': best_score_mean,
                    }, step=episode)
                                        
                    self.update_loss = []
                    
                    # Stop training if the max episode limit is reached
                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    
                    # Decay epsilon
                    self.epsilon = max(self.epsilon * self.eps_decay, 0.01)
                    torch.save(self.dnnetwork.state_dict(), "model_rainbow_last.pth")

                    
    # Loss calculation for training
    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = batch

        states = torch.FloatTensor(states).to(device=self.dnnetwork.device)
        next_states = torch.FloatTensor(next_states).to(device=self.dnnetwork.device)
        rewards = torch.FloatTensor(rewards).to(device=self.dnnetwork.device)
        actions = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(device=self.dnnetwork.device)
        dones = torch.BoolTensor(dones).to(device=self.dnnetwork.device)

        if states.dim() == 3:
            states = states.unsqueeze(1)  # Add channel dimension if missing
        if next_states.dim() == 3:
            next_states = next_states.unsqueeze(1)  # Add channel dimension if missing

        # Q-values for the current state-action pairs
        current_qvals = torch.gather(self.dnnetwork.get_qvals(states), 1, actions)

        with torch.no_grad():
            # Action selection using the main network
            next_actions = torch.argmax(self.dnnetwork.get_qvals(next_states), dim=1, keepdim=True)

            # Evaluation of the selected actions using the target network
            next_q_values = torch.gather(self.target_network.get_qvals(next_states), 1, next_actions).squeeze(1)

            # Zero out the next Q-values for terminal states
            next_q_values[dones] = 0.0

            # Compute the expected Q-values
            expected_q_values = rewards + self.gamma * next_q_values

        # Calculate loss
        loss = torch.nn.MSELoss()(current_qvals.squeeze(), expected_q_values)
        return loss



    # Update the main network
    def update(self):
        self.dnnetwork.optimizer.zero_grad()
        batch = self.buffer.sample_batch(self.batch_size)
        loss = self.calculate_loss(batch)
        self.update_loss.append(loss.item())
        loss.backward()
        self.dnnetwork.optimizer.step()
        return loss.item()

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, done, next_state):
        self.buffer.append((state, action, reward, done, next_state))

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)
        
        states = np.stack(states)
        next_states = np.stack(next_states)
        return (
            states.astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            next_states.astype(np.float32)
        )

    def burn_in_capacity(self):
        return len(self.buffer) / self.capacity

# Define prioritized replay buffer
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6):
        super().__init__(capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.pos = 0

    def append(self, state, action, reward, done, next_state):
        max_priority = self.priorities.max() if self.buffer else 1.0
        super().append(state, action, reward, done, next_state)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample_batch(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Replay buffer is empty!")

        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, dones, next_states = zip(*samples)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            np.stack(next_states).astype(np.float32),
            indices,
            weights
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

# Define noisy layer for Rainbow DQN
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        self.weight_epsilon.copy_(self.f(epsilon_out).ger(self.f(epsilon_in)))
        self.bias_epsilon.copy_(self.f(epsilon_out))

    @staticmethod
    def f(x):
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

# Define Rainbow DQN with noisy layers and dueling architecture
class RainbowDQN(DDQN):
    def __init__(self, env, learning_rate=1e-3, device='cpu', atoms=51, Vmin=-10, Vmax=10):
        super().__init__(env, learning_rate, device)
        self.atoms = atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.support = torch.linspace(Vmin, Vmax, atoms).to(device)
        self.delta_z = (Vmax - Vmin) / (atoms - 1)

        linear_input_size = self.fc[0].in_features
        self.advantage = nn.Sequential(
            NoisyLinear(linear_input_size, HIDDEN_SIZE),
            nn.ReLU(),
            NoisyLinear(HIDDEN_SIZE, n_actions * atoms)
        )
        self.value = nn.Sequential(
            NoisyLinear(linear_input_size, HIDDEN_SIZE),
            nn.ReLU(),
            NoisyLinear(HIDDEN_SIZE, atoms)
        )

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        advantage = self.advantage(x).view(-1, n_actions, self.atoms)
        value = self.value(x).view(-1, 1, self.atoms)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.softmax(q_values, dim=2)

    def get_qvals(self, state):
        with torch.no_grad():
            distribution = self(state)
            qvals = torch.sum(distribution * self.support, dim=2)
        return qvals
    
def get_gameplays_and_add_to_buffer (path):
    with open(path, "rb") as f:
        gameplay_data = pickle.load(f)
    
    observations = gameplay_data["observations"]
    actions = gameplay_data["actions"]
    rewards = gameplay_data["rewards"]
    print(f"Observations: {len(observations)}")
    
    for i in range(len(observations) - 1):
        state = preprocess_frame(observations[i])
        next_state = preprocess_frame(observations[i + 1])
        action = actions[i]
        reward = rewards[i]
        done = i == len(observations) - 1
        
        buffer.append(state, action, reward, done, next_state)    
    
buffer = PrioritizedReplayBuffer(capacity=MEMORY_SIZE)

for i in range(15):
    get_gameplays_and_add_to_buffer(f"manual_gameplays/skiing_gameplay_data_{i}.pkl")


# Initialize Rainbow DQN and prioritized replay buffer
wandb.login()
wandb.init(project="Prova", config={
    "lr": LR,
    "MEMORY_SIZE": MEMORY_SIZE,
    "MAX_EPISODES": MAX_EPISODES,
    "EPSILON": EPSILON,
    "EPSILON_DECAY": EPSILON_DECAY,
    "GAMMA": GAMMA,
    "BATCH_SIZE": BATCH_SIZE,
    "BURN_IN": BURN_IN,
    "DNN_UPD": DNN_UPD,
    "DNN_SYNC": DNN_SYNC
})

rainbow_net = RainbowDQN(env, learning_rate=LR, device=DEVICE)
agent = DDQNAgent(env, rainbow_net, buffer, epsilon=EPSILON, eps_decay=EPSILON_DECAY, batch_size=BATCH_SIZE)

agent.train(gamma=GAMMA, max_episodes=MAX_EPISODES, 
              batch_size=BATCH_SIZE,
              dnn_update_frequency=DNN_UPD,
              dnn_sync_frequency=DNN_SYNC)
