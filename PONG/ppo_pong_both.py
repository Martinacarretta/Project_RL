import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import cv2
import ale_py
import wandb
from wandb.integration.sb3 import WandbCallback

import supersuit as ss
import torch
import ppo_pong_constants as constants

############## LEFT PLAYER WRAPPER
class TransposeAndFlipObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, flip=False):
        super().__init__(env)
        original_space = env.observation_space
        self.flip = flip
        self.observation_space = gym.spaces.Box(
            low=original_space.low.transpose(2, 0, 1),
            high=original_space.high.transpose(2, 0, 1),
            dtype=original_space.dtype
        )

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        if self.flip:
            observation = np.flip(observation, axis=2)  # Flip horizontally along the width
        return observation
    
class FlipRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, flip=False):
        super().__init__(env)
        self.flip = flip

    def reward(self, reward):
        return -reward if self.flip else reward  # Flip rewards for left player

# Function to create the environment
def create_env(player_side="right"):
    env = gym.make("PongNoFrameskip-v4")
    env = Monitor(env)  # Place Monitor wrapper before VecEnv wrappers
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)

    # Apply TransposeAndFlipObservationWrapper based on player side
    if player_side == "left":
        env = TransposeAndFlipObservationWrapper(env, flip=True)
        env = FlipRewardWrapper(env, flip=True)  # Flip rewards for the left player

    elif player_side == "right":
        env = TransposeAndFlipObservationWrapper(env, flip=False)

    return DummyVecEnv([lambda: env])

# Initialize Weights & Biases (WandB)
run = wandb.init(
    project="PONG prova",
    config={
        "env_id": constants.env_id,
        'Policy': constants.policy,
        "algorithm": constants.algorithm,
        "learning_rate": constants.learning_rate,  
        "gamma": constants.gamma,
        "gae_lambda": constants.gae_lambda,
        "n_steps": constants.n_steps,  
        "ent_coef": constants.ent_coef, 
        "vf_coef": constants.vf_coef,
        "clip_range": constants.clip_range, 
        "clip_range_vf": constants.clip_range_vf,
        "n_epochs": constants.n_epochs, 
        "batch_size": constants.batch_size, 
        "max_grad_norm": constants.max_grad_norm,
        "total_timesteps": constants.total_timesteps,
        "model_name": constants.model_name,
        "export_path": constants.export_path,
        "videos_path": constants.videos_path,
    },
    sync_tensorboard=True,
    save_code=True,
)

# Create environments for both sides
right_player_env = create_env(player_side="right")
left_player_env = create_env(player_side="left")

# Train self-play
def self_play_training(model, total_timesteps, switch_freq):
    steps = 0
    current_env = 0
    player_sides = ["right", "left"]

    # Track rewards and episodes
    rewards_tracker = {side: deque(maxlen=100) for side in player_sides}
    episode_counts = {side: 0 for side in player_sides}

    while steps < total_timesteps:
        timesteps = min(switch_freq, total_timesteps - steps)
        current_env_side = player_sides[current_env]

        # Train the model
        model.set_env(right_player_env if current_env == 0 else left_player_env)
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            callback=None  # Disable excessive logging
        )

        # Evaluate performance
        for env, side in zip([right_player_env, left_player_env], player_sides):
            obs = env.reset()
            done = False
            episode_reward = 0
            steps_in_episode = 0
            while not done:
                action = model.predict(obs, deterministic=True)[0]
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                steps_in_episode += 1
                steps += 1  # Increment the global step count
            
            # Ensure reward is scalar
            if isinstance(episode_reward, np.ndarray):
                episode_reward = episode_reward.item()


            rewards_tracker[side].append(episode_reward)
            episode_counts[side] += 1
            mean_reward = np.mean(rewards_tracker[side])

            # Log stats to WandB
            wandb.log({
                f"{side}_player_reward": episode_reward,
                f"{side}_mean_reward_100_episodes": mean_reward,
                "total_steps": steps,
                "current_env": side,
                "episodes_completed": episode_counts[side],
            })

            print(
                f"Steps: {steps} | Side: {side} | Episodes: {episode_counts[side]} | "
                f"Reward: {episode_reward:.2f} | Mean Reward (last 100): {mean_reward:.2f}"
            )

        steps += timesteps
        current_env = 1 - current_env  # Switch environment

# Define the PPO model
model = PPO(
    "CnnPolicy",  
    right_player_env,  # Start with the right player environment
    learning_rate = constants.learning_rate,
    gamma = constants.gamma,
    gae_lambda = constants.gae_lambda,
    n_steps = constants.n_steps,
    ent_coef = constants.ent_coef,
    vf_coef = constants.vf_coef,
    clip_range_vf = constants.clip_range_vf,
    clip_range = constants.clip_range,
    n_epochs = constants.n_epochs,
    batch_size = constants.batch_size,
    max_grad_norm = constants.max_grad_norm,
    verbose=0,  # Silence excessive output
    tensorboard_log=f"runs/{run.id}",
    device='cuda',
    policy_kwargs={'normalize_images': False},
)

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=2000,  # Save the model every 2000 steps
    save_path='./checkpoints/',  # Directory to save the checkpoints
    name_prefix="ppo_pong",  # Prefix for the checkpoint filenames
)
callback_list = CallbackList([WandbCallback(verbose=0), checkpoint_callback])

# Train with self-play
print("Training with self-play...")
print("Is CUDA available:", torch.cuda.is_available())
self_play_training(model, constants.total_timesteps, switch_freq=1)

# Save the trained model
model_path = os.path.join(constants.export_path, constants.model_name)
model.save(model_path)
wandb.save(model_path + ".zip")
wandb.finish()

# Close environments
right_player_env.close()
left_player_env.close()
