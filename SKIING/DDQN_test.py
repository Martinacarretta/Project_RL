import gymnasium as gym
import ale_py
import shimmy

import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn

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
NUM_ACTIONS = 3  # Number of actions the agent can take
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

#test:

net = DDQN (env, learning_rate=LR, device=DEVICE)
pretrained_model_path = "model_DDQN_manual.pth"

if os.path.exists(pretrained_model_path):
    print(f"Loading model from {pretrained_model_path}")
    net.load_state_dict(torch.load(pretrained_model_path, map_location=DEVICE))
    print("model loaded successfully.")
else:
    print(f"model not found at {pretrained_model_path}.")
    
# 2. Test the model for 10 episodes and calculate the average reward and save the rewards to plot them 
# Save also the gifs from the rendered frames in a folder called "gifs_test_DDQN"

n_episodes = 100
rewards = []
for i in range(n_episodes):
    state = preprocess_frame(env.reset()[0])
    state = torch.FloatTensor(state).to(DEVICE)  # Move to the correct device

    done = False
    total_reward = 0
    frames = []
    frame_num = 0
    while not done:
        frame_num += 1
        action = net.get_action(state, epsilon=0.05)
        next_state, reward, terminated, truncated, _, = env.step(action)
        next_state = preprocess_frame(next_state)
        next_state = torch.FloatTensor(next_state).to(DEVICE)  # Ensure compatibility

        total_reward += reward
        state = next_state
        frame = env.render()
        frames.append(frame)
        if frame_num >= 2000:
            break
    rewards.append(total_reward)
    print(f"Episode {i + 1} Reward: {total_reward}")
    
    # Save the frames as a gif
    frames = [Image.fromarray(frame) for frame in frames]
    # Check if the directory exists, if not create it
    os.makedirs("gifs_test_DDQN", exist_ok=True)
    frames[0].save(f"gifs_test_DDQN/episode_{i + 1}.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    
print(f"Average reward over {n_episodes} episodes: {np.mean(rewards)}")

# 3. Save the plot of the rewards in the folder used for the gifs
# plot the mean 
os.makedirs("gifs_test_DDQN", exist_ok=True)
plt.plot(rewards)
plt.axhline(np.mean(rewards), color='r', linestyle='--', label=f"Mean Reward: {np.mean(rewards)}")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards test Plot")
plt.savefig("gifs_test_DDQN/rewards_plot.png")
plt.close()
print("Rewards plot saved successfully.")