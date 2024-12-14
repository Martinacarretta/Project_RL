import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import cv2
import ale_py
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor

def make_env(env_id):
    def _env():
        env = gym.make(env_id)
        env = Monitor(env, allow_early_resets=True)  
        return env
    return _env

#512

run = wandb.init(
    project="A2C skiing",
    config={
        "env_id": "ALE/Skiing-v5",
        'Policy':"CnnPolicy",
        "algorithm": "A2C",
        "learning_rate": 1e-4,  #0.001  #lambda f : f * 2.5e-4 is a learning rate schedule
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_steps": 1024,  #128 [512,1024,2048,4096]
        "ent_coef": 0.01, #0.01 #exploration coef  #PROVAR 0 AMB 'CnnPolicy'
        "vf_coef": 0.5,
        "clip_range": 0.2, #[0.1-0.3]
        "clip_range_vf": 1,
        "n_epochs": 6, #4
        "batch_size": 1024, # between 32-512 in discrete action spaces
        "max_grad_norm": 0.4,
        "total_timesteps": 10000000,
        "model_name": "A2C_skiing",
        "export_path": "./exports/",
        "videos_path": "./videos/",
    },
    sync_tensorboard=True,
    save_code=True,
)


env_id = "ALE/Skiing-v5"  # Pac-Man environment ID
env = DummyVecEnv([make_env(env_id) for i in range(4)])  # Create 8 parallel environments

# Define the A2C model
model = A2C(
    "CnnPolicy",  # Convolutional Neural Network policy  #CnnPolicy
    env,
    learning_rate=wandb.config.learning_rate,
    gamma=wandb.config.gamma,
    gae_lambda=wandb.config.gae_lambda,
    n_steps=wandb.config.n_steps,
    ent_coef=wandb.config.ent_coef,
    vf_coef=wandb.config.vf_coef,
    max_grad_norm=wandb.config.max_grad_norm,
    verbose=2,
    tensorboard_log=f"runs/{run.id}",
)


env_id = "ALE/Skiing-v5"  
eval_env = DummyVecEnv([make_env(env_id) for i in range(1)])  


eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=False, render=False)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,  # Save the model every 1000 steps
    save_path='./checkpoints/',  # Directory to save the checkpoints
    name_prefix="A2C_skiing"  # Prefix for the checkpoint filenames
)

callback_list = CallbackList([WandbCallback(verbose=2), eval_callback, checkpoint_callback])

# Train the model
print("Training...")
model.learn(total_timesteps=wandb.config.total_timesteps, callback=callback_list)

# Save the trained model
model_path = os.path.join(wandb.config["export_path"], wandb.config["model_name"])
model.save(model_path)
wandb.save(model_path + ".zip")
wandb.finish()

# Close the training environment
env.close()