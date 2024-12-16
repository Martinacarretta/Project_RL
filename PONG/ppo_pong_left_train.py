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

##############LEFT PLAYER

class TransposeAndFlipObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        original_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=original_space.low.transpose(2, 0, 1),
            high=original_space.high.transpose(2, 0, 1),
            dtype=original_space.dtype
        )

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        # Flip horizontally along the width (axis=2)
        return np.flip(observation, axis=2)  



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
env = Monitor(env)  # Place Monitor wrapper before VecEnv wrappers
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)
env = ss.dtype_v0(env, dtype=np.float32)
env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
env = TransposeAndFlipObservationWrapper(env) ############
print(f"shape and range of the observation space: {env.observation_space.shape}, {env.observation_space.low[0][0][0]}, {env.observation_space.high[0][0][0]}")
env = DummyVecEnv([lambda: env]) #####################

print("Observation space details:", env.observation_space)



# Define the PPO model
model = PPO(
    "CnnPolicy",  
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
    device = 'cuda',
    policy_kwargs={'normalize_images': False},
)

eval_env = gym.make("PongNoFrameskip-v4")
eval_env = Monitor(eval_env)  # Place Monitor wrapper before VecEnv wrappers
eval_env = ss.color_reduction_v0(eval_env, mode="B")
eval_env = ss.resize_v1(eval_env, x_size=84, y_size=84)
eval_env = ss.frame_stack_v1(eval_env, 4)
eval_env = ss.dtype_v0(eval_env, dtype=np.float32)
eval_env = ss.normalize_obs_v0(eval_env, env_min=0, env_max=1)
eval_env = TransposeAndFlipObservationWrapper(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])  # Wrap in DummyVecEnv for vectorization


eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1000,
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

