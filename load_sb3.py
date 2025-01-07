# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


interm_dir = "./logs/intermediate_models/"

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training

###############################################################################################################
#00FFFF Setting 1:
LEARNING_ALG = "SAC"
# target_dir   = '121024143554'   #00FF00 # path to saved models, i.e. interm_dir + '102824115106'
target_dir   = '2411101145_pd_SAC_NoNoise_FLAT_Local_new'
#00FFFF Setting 2:
env_config = { "motor_control_mode":     "PD",
               "task_env":               "FWD_LOCOMOTION", #"FWD_LOCOMOTION", 
               "observation_space_mode": "DEFAULT",  # "DEFAULT", "LR_COURSE_OBS",
               "terrain":                None, #SLOPES', #"SLOPES", #"SLOPES", #"RANDOM",  
               "render":                 False, #True,
               "record_video":           False, #False,
               "add_noise":              False,
               "EPISODE_LENGTH":         10,
             } 

# env_config['competition_env'] = True
###############################################################################################################
log_dir = interm_dir + target_dir

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)


# Plot results
# monitor_results = load_results(log_dir)
# print(monitor_results)
# plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
# plt.show() 


# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)   # Environment Setting 环境参数
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)

env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

################################################################################################################
# #0000FF TODO initialize arrays to save data from simulation
TotalTime  = 5
TIME_STEP  = env.envs[0].env._time_step * 10  # 0.001 * 10
TEST_STEPS = int(TotalTime / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# Data record
cpg_states           = np.zeros((TEST_STEPS, 4, 4))  # Shape: [time_steps, 4_legs, 4_states (r,theta,r_dot,theta_dot)]
foot_positions_real  = np.zeros((TEST_STEPS, 4, 3))  # Shape: [time_steps, 4_legs, 3_coordinates]
foot_positions_des   = np.zeros((TEST_STEPS, 4, 3))  # Shape: [time_steps, 4_legs, 3_coordinates]
foot_angles_real     = np.zeros((TEST_STEPS, 4, 3))  # Shape: [time_steps, 4_legs, 3_angles]
foot_angles_des      = np.zeros((TEST_STEPS, 4, 3))  # Shape: [time_steps, 4_legs, 3_angles]
base_positions       = np.zeros((TEST_STEPS, 3))     # Shape: [time_steps, 3_coordinates]
base_velocities      = np.zeros((TEST_STEPS, 3))     # Shape: [time_steps, 3_coordinates]
base_RollPitchYaw    = np.zeros((TEST_STEPS, 3))     # Shape: [time_steps, RollPitchYaw]
energy_list          = np.zeros(TEST_STEPS)          # Shape: [time_steps]
################################################################################################################

for i in range(TEST_STEPS):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? (#0000FF TODO: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0

    # #0000FF TODO save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    base_positions [i, :] = env.envs[0].env.robot.GetBasePosition()
    base_velocities[i, :] = env.envs[0].env.robot.GetBaseLinearVelocity()
    base_RollPitchYaw[i, :] = env.envs[0].env.robot.GetBaseOrientationRollPitchYaw()
    energy = 0
    for tau,vel in zip (env.envs[0].env._dt_motor_torques,env.envs[0].env._dt_motor_velocities):
        energy += np.abs(np.dot(tau,vel)) * env.envs[0].env._time_step
    energy_list[i] = energy

# #0000FF TODO make plots:
from functions.plot import *
# plot_base_velocity(t, base_velocities, figsize=(10, 3), window_size=500) 
# plot_RollPitch(t, base_RollPitchYaw, figsize=(10, 6))

# 示例：传入要绘制的变量
data_list = ['energy', 'pitch', 'x_velocity']  # 可以是 ['energy', 'roll', 'pitch', 'yaw', 'x_velocity']
plot_data(t, data_list, base_positions, base_velocities, base_RollPitchYaw, energy_list)
