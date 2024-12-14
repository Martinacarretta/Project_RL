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
import pickle
import os


# Create the environment
env = gym.make("ALE/Skiing-v5", render_mode="rgb_array")
# env = gym.make("PongNoFrameskip-v4")

# Set device for CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
HIDDEN_SIZE = 256
LEARNING_RATE = 0.0001
GAMMA = 0.99
HORIZON = 15000
MAX_TRAJECTORIES = 5000 #10000

DEMONSTRATION_PROB = 0.2  # Probability of using demonstration data

# Image preprocessing
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Normalize
    normalized = (resized / 255.0) * 2 - 1
    return normalized

# Define a more complex policy model for image input
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
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

# Initialize model and optimizer
n_actions = env.action_space.n
model = PolicyNetwork(input_shape=(1, 84, 84), n_actions=n_actions).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Function to calculate discounted rewards
def calculate_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    running_add = 0
    for reward in reversed(rewards):
        running_add = running_add * gamma + reward
        discounted_rewards.insert(0, running_add)
    return np.array(discounted_rewards)

# Function to standardize rewards
def standardize_rewards(rewards):
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

################################################################################################
# Load gameplay data
with open("skiing_gameplay_data.pkl", "rb") as f:
    gameplay_data = pickle.load(f)
    
gameplay_observations = gameplay_data["observations"]
gameplay_actions = gameplay_data["actions"]
gameplay_rewards = gameplay_data["rewards"]
    
print(f"Observations: {len(gameplay_data)}")
    
    
def train_with_demonstration(demonstration_data, model, optimizer):
    state, action, reward = demonstration_data
    state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)  # Adjust if necessary
    action_tensor = torch.LongTensor([action]).to(DEVICE)
    reward_tensor = torch.FloatTensor([reward]).to(DEVICE)
    
    # Calculate action probabilities from the policy model
    action_probs = model(state_tensor)
    selected_log_probs = torch.log(action_probs.gather(1, action_tensor.unsqueeze(1)).squeeze())
    
    # Calculate loss
    demonstration_loss = -torch.mean(selected_log_probs * reward_tensor)
    return demonstration_loss


# Wandb setup
wandb.init(project="Skiing-REINFORCE", config={
    "HIDDEN_SIZE": HIDDEN_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "GAMMA": GAMMA,
    "HORIZON": HORIZON,
    "MAX_TRAJECTORIES": MAX_TRAJECTORIES
})

# Training loop
scores = []
mean_rewards = deque(maxlen=100)
losses = []
best_score_mean = -np.inf
best_score_episode = -np.inf

for trajectory in range(MAX_TRAJECTORIES):
    length = len(gameplay_observations)
    print(f"Length: {length}")
    
    if random.random() < DEMONSTRATION_PROB:
        random_index = random.randint(0, length - 1)
        demonstration_data = (gameplay_observations[random_index], gameplay_actions[random_index], gameplay_rewards[random_index])
        print(f"Using demonstration data from index {random_index}")
        demonstration_loss = train_with_demonstration(demonstration_data, model, optimizer)
        loss += demonstration_loss

    images = []  # Record rendered frames
    # Reset environment
    state, _ = env.reset()
    state = preprocess_frame(state)
    
    transitions = []
    total_reward = 0
    frames = 0
    # Collect trajectory
    for t in range(HORIZON):
        frames += 1
        # Prepare state tensor - MODIFIED HERE
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Get action probabilities
        action_probs = model(state_tensor)
        
        epsilon = max(0.1, 1 - trajectory / 100)  # Decay exploration rate

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(action_probs).item()
        
        img = env.render()
        if img is not None:  # Check if the render method returns a valid image
            images.append(Image.fromarray(img))


        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # print(f"Reward: {reward}")
        next_state = preprocess_frame(next_state)

        # Store transition
        transitions.append((state, action, reward))
        total_reward += reward
        
        state = next_state

        if terminated or truncated:
            break

    # Process rewards
    states, actions, rewards = zip(*transitions)
    
    # MODIFIED HERE - ensure states are processed correctly
    states = np.array(states)
    states = states[..., np.newaxis]  # add channel dimension
    states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(DEVICE)
    
    actions = torch.LongTensor(actions).to(DEVICE)
    
    discounted_rewards = calculate_discounted_rewards(rewards, GAMMA)
    # normalized_rewards = standardize_rewards(discounted_rewards)
    rewards = torch.FloatTensor(discounted_rewards).to(DEVICE)

    # Calculate loss
    action_probs = model(states)
    selected_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
    loss = -torch.mean(selected_log_probs * rewards)        

    # Update policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    scores.append(total_reward)
    mean_rewards.append(total_reward)
    losses.append(loss.item())

    # Print progress
    # if trajectory % 50 == 0:
    print(f"Episode {trajectory}\tMean Reward: {np.mean(mean_rewards):.2f}")
        
    wandb.log({
        'episode': trajectory,
        'episode_reward': total_reward,
        'mean_rewards': np.mean(mean_rewards),
        'loss': loss.item(), 
        'frames': frames, 
        'best score mean': best_score_mean,
        'best score episode': best_score_episode
    })
    
    if total_reward > best_score_episode:
        best_score_episode = total_reward
        torch.save(model.state_dict(), "model_reinforce_best_ep_manual.pth")
        if images:
            # Save gif in a folder named "gifs_reinforce_manual" without replacing any of the stored gifs
            # Ensure the folder exists
            folder_path = "gifs_reinforce_manual"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Save the GIF
            images[0].save(f"{folder_path}/episode_{trajectory}_reward_{total_reward}.gif", save_all=True, append_images=images[1:], loop=0)
            print("New best episode score! Video saved.")
    
    if total_reward > best_score_mean:
        best_score_mean = np.mean(mean_rewards)
        torch.save(model.state_dict(), "model_reinforce_manual.pth")

    # Optional early stopping condition
    if trajectory > 100 and np.mean(mean_rewards) > -1000:
        print(f"Solved in {trajectory} episodes!")
        break

env.close()
wandb.finish()