import gymnasium as gym
import ale_py
import shimmy

import numpy as np
import matplotlib.pyplot as plt
import random

import cv2

from collections import namedtuple, deque
from copy import deepcopy, copy
import timeit
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import wandb
import os


# Create the environment
env = gym.make("ALE/Skiing-v5", render_mode="rgb_array")
# env = gym.make("PongNoFrameskip-v4")

# Set device for CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 256

# Image preprocessing
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Normalize
    normalized = (resized / 255.0) * 2 - 1
    return normalized

n_actions = env.action_space.n

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the output from conv layers
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, n_actions)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.softmax(x, dim=-1)

# Replace nn.Sequential with PolicyNetwork
model = PolicyNetwork(input_shape=(1, 84, 84), n_actions=n_actions).to(DEVICE)

# Load the trained model
model.load_state_dict(torch.load("model_reinforce.pth"))
model.eval()  # Set the model to evaluation mode


# Variables to keep track of scores
test_scores = []

episodes_rewards = []

# Test for 100 episodes
for episode in range(100):
    images = []
    state, _ = env.reset()
    state = preprocess_frame(state)
    total_reward = 0
    done = False
    rewards = []
    
    while not done:
        # Prepare state tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Get action probabilities
        with torch.no_grad():  # Disable gradient computation for testing
            action_probs = model(state_tensor)
            action = torch.argmax(action_probs).item()
        
        # Take action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        next_state = preprocess_frame(next_state)
        #print("Reward: ", reward)
        
        img = env.render()
        images.append(img)
        
        total_reward += reward
        state = next_state

        # Check if the episode is over
        done = terminated or truncated
    
    episodes_rewards.append(rewards)

    # Append the total reward for this episode
    test_scores.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    frames = [Image.fromarray(frame) for frame in images]
    os.makedirs("gifs_test_reinforce", exist_ok=True)
    frames[0].save(f"gifs_test_reinforce/episode_{episode}.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
        
print(episodes_rewards[1])
print(len(episodes_rewards[1]))
if episodes_rewards[0] == episodes_rewards[1]:
    if episodes_rewards[1] == episodes_rewards[2]:
        if episodes_rewards[2] == episodes_rewards[3]:
            print("Same")
        else:
            print("Not same")
                
# Calculate and print statistics
average_score = np.mean(test_scores)
std_dev_score = np.std(test_scores)
print(f"Test completed over 100 episodes.")
print(f"Average Score: {average_score:.2f}, Standard Deviation: {std_dev_score:.2f}")


os.makedirs("gifs_test_reinforce", exist_ok=True)
plt.plot(test_scores)
plt.axhline(average_score, color='r', linestyle='--', label=f"Mean Reward: {average_score}")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards test Plot")
plt.savefig("gifs_test_reinforce/rewards_plot.png")
plt.close()
print("Rewards plot saved successfully.")