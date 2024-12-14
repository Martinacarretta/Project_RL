import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Gymnasium and ALE
import gymnasium as gym
import ale_py

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

# Wandb Integration
import wandb
from wandb.integration.sb3 import WandbCallback

# Custom constants
import ppo_pacman_constants as constants

def make_env(env_id, render_mode=None):
    def _env():
        env = gym.make(env_id, render_mode=render_mode)
        env = Monitor(env, allow_early_resets=True)  
        return env
    return _env

run = wandb.init(
    project="PPO Pacman",
    config={
        "env_id": constants.env_id,
        'Policy':constants.policy,
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

env_id = "ALE/Pacman-v5"  # Pac-Man environment ID
env = DummyVecEnv([make_env(env_id) for i in range(8)])  

# Define the PPO model
model = PPO(
    "CnnPolicy",  # Convolutional Neural Network policy 
    env,
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
    verbose = 2,
    tensorboard_log = f"runs/{run.id}",
    device = 'cuda'
)

eval_env = DummyVecEnv([make_env(env_id) for i in range(1)])  


eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1000,
                             deterministic=False, render=False)

checkpoint_callback = CheckpointCallback(
    save_freq=2000,  # Save the model every 2000 steps
    save_path='./checkpoints/',  # Directory to save the checkpoints
    name_prefix="ppo_pacman",  # Prefix for the checkpoint filenames
)

callback_list = CallbackList([WandbCallback(verbose=2), eval_callback, checkpoint_callback])

# Train the model
print("Training...")
model.learn(total_timesteps=constants.total_timesteps, callback=callback_list)

# Save the trained model
model_path = os.path.join(constants.export_path, constants.model_name)
model.save(model_path)
wandb.save(model_path + ".zip")
wandb.finish()

# Close the training environment
env.close()

