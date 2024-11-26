import sys
import os

# 将上一级目录加入搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import load_results

from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results
from utils.file_utils import get_latest_model
from utils.utils import ts2xy

# 定义 class 类
class QuadrupedSimulation:
    def __init__(self, algorithm="SAC", log_dir="./logs/intermediate_models/"):
        """
        Initialize the simulation class.

        Parameters:
        - algorithm (str): Learning algorithm ("PPO" or "SAC").
        - log_dir (str): Directory path where models and logs are saved.
        """
        self.algorithm = algorithm
        self.log_dir = log_dir
        self.env_config = {}
        self.model = None
        self.env = None
        self.stats_path = None
        self.model_name = None

    def set_env_config(self, render = True, 
                       record_video = False, 
                       motor_control_mode="CPG",                  # motor control mode
                       task_env="FWD_LOCOMOTION",                 # task
                       observation_space_mode="LR_COURSE_OBS",    # observation space
                       terrain    = None,                         # terrain
                       add_noise  = False,                        # moise
                       ):   

        self.env_config = {
            "render":                 render,
            "record_video":           record_video,
            "motor_control_mode":     motor_control_mode,
            "task_env":               task_env,
            "observation_space_mode": observation_space_mode,
            "terrain":                terrain,
            "add_noise":              add_noise,
        }

    def load_plots(self):
        """
        Load the model and environment based on the provided configuration.
        """
        # Locate the latest model and stats path
        self.stats_path = os.path.join(self.log_dir, "vec_normalize.pkl")
        self.model_name = get_latest_model(self.log_dir)

        # print(monitor_results)
        plot_results([self.log_dir], 10e10, 'timesteps', self.algorithm + ' ')   #00FF00 画图函数
        plt.show()


    def load_and_run(self, steps=2000, deterministic=False):
        """
        Load the model, set up the environment, and run the simulation.

        Parameters:
        - steps (int): Number of steps to simulate.
        - deterministic (bool): Whether to use deterministic actions during testing.
        """
        # Locate the latest model and stats path
        self.stats_path = os.path.join(self.log_dir, "vec_normalize.pkl")
        self.model_name = get_latest_model(self.log_dir)

        # Create the environment
        env = lambda: QuadrupedGymEnv(**self.env_config)
        self.env = make_vec_env(env, n_envs=1)
        self.env = VecNormalize.load(self.stats_path, self.env)
        self.env.training = False
        self.env.norm_reward = False

        # Load the model
        if self.algorithm == "PPO":
            self.model = PPO.load(self.model_name, self.env)
        elif self.algorithm == "SAC":
            self.model = SAC.load(self.model_name, self.env)
        print(f"\nLoaded model: {self.model_name}\n")

        # Run simulation
        obs = self.env.reset()
        episode_reward = 0

        for i in range(steps):
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, info = self.env.step(action)
            episode_reward += rewards
            if dones:
                print('Episode reward:', episode_reward)
                print('Final base position:', info[0]['base_pos'])
                episode_reward = 0


# 示例使用
if __name__ == "__main__":
    sim = QuadrupedSimulation(algorithm="SAC", log_dir="./logs/intermediate_models/112624155608")
    sim.set_env_config(render=True, 
                       record_video=False, 
                       add_noise=False,
                       motor_control_mode="CPG", 
                       task_env="FWD_LOCOMOTION", 
                       observation_space_mode="LR_COURSE_OBS")
    sim.load_plots()   # plot rewards
    sim.load_and_run()
    sim.run_simulation(steps=2000, deterministic=False)
