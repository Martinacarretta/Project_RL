import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import cv2
import ale_py
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from PIL import Image
import supersuit as ss
import torch

import ppo_pong_constants as constants


run = wandb.init(
    project="PPO Pong",
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


env = gym.make("PongNoFrameskip-v4")
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)
env = ss.dtype_v0(env, dtype=np.float32)
env = ss.normalize_obs_v0(env, env_min=0, env_max=1)

# env = gym.wrappers.AtariPreprocessing(
#     env, 
#     frame_skip=4, 
#     grayscale_obs=False,  
#     scale_obs=False  
#     )
# env = Monitor(env, allow_early_resets=True) 
# env = ss.color_reduction_v0(env, mode='B')  # Reduces the color of frames to black and white
# env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize the observation space
# env = ss.frame_stack_v1(env, 4)  # Stack 4 frames together
# env = ss.dtype_v0(env, dtype='uint8')
# env = DummyVecEnv([lambda: env])
# env = VecTransposeImage(env)  # Convert channel-last to channel-first (C, H, W)


# print("Observation space type:", type(env.observation_space))
# print("Observation space details:", env.observation_space)

# obs = env.reset()
# print("Sample observation shape:", obs.shape)

# Define the PPO model
model = PPO(
    "MlpPolicy",  
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

eval_env = gym.make("PongNoFrameskip-v4")
eval_env = ss.color_reduction_v0(eval_env, mode="B")
eval_env = ss.resize_v1(eval_env, x_size=84, y_size=84)
eval_env = ss.frame_stack_v1(eval_env, 4)
eval_env = ss.dtype_v0(eval_env, dtype=np.float32)
eval_env = ss.normalize_obs_v0(eval_env, env_min=0, env_max=1)
# eval_env = gym.wrappers.AtariPreprocessing(
#     eval_env, 
#     frame_skip=4, 
#     grayscale_obs=False,  
#     scale_obs=False  
#     )

# eval_env = Monitor(eval_env, allow_early_resets=True)
# eval_env = ss.color_reduction_v0(eval_env, mode='B')  # Grayscale
# eval_env = ss.resize_v1(eval_env, x_size=84, y_size=84)  # Resize
# eval_env = ss.frame_stack_v1(eval_env, 4)  # Stack 4 frames
# eval_env = ss.dtype_v0(eval_env, dtype='uint8')  # Correct dtype
# eval_env = DummyVecEnv([lambda: eval_env])  # VecEnv wrapping
# eval_env = VecTransposeImage(eval_env)  # Channel-first format



eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=False, render=False)

checkpoint_callback = CheckpointCallback(
    save_freq=2000,  # Save the model every 2000 steps
    save_path='./checkpoints/',  # Directory to save the checkpoints
    name_prefix="ppo_pong",  # Prefix for the checkpoint filenames
)

callback_list = CallbackList([WandbCallback(verbose=2), eval_callback, checkpoint_callback])

# Train the model
print("Training...")
print("Is CUDA available:", torch.cuda.is_available())
model.learn(total_timesteps=constants.total_timesteps, callback=callback_list)

# Save the trained model
model_path = os.path.join(constants.export_path, constants.model_name)
model.save(model_path)
wandb.save(model_path + ".zip")
wandb.finish()

# Close the training environment
env.close()

