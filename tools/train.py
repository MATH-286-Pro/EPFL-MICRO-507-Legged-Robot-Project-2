import sys
import os

# 将上一级目录加入搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import torch  # import PyTorch to detect GPU
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from utils.utils import CheckpointCallback
from utils.file_utils import get_latest_model
from env.quadruped_gym_env import QuadrupedGymEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from IPython.display import clear_output


class QuadrupedTrainer:
    def __init__(self, algorithm = "PPO", 
                 num_envs        = 1, 
                 load_last_train = False,  
                 load_dir        = "None"
                 ):
        """
        Initialize the trainer class.

        Parameters:
        - algorithm (str): Learning algorithm ("PPO" or "SAC").
        - num_envs (int): Number of environments for parallel training.
        - load_last_train (bool): Whether to load a pre-trained model.
        - log_dir (str): Directory to save logs and models.
        """
        self.algorithm = algorithm
        self.num_envs = num_envs
        self.load_last_train = load_last_train
        # self.log_dir = log_dir
        self.env_configs = {}
        self.model = None
        self.env = None
        self.SAVE_PATH = None

        self.base_log_dir = "./logs/intermediate_models/"
        self.log_dir = f"{self.base_log_dir}{load_dir}" if load_dir else self.base_log_dir

        # Auto detect GPU
        self.gpu_arg = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.gpu_arg}")

        self.checkpoint_callback = None

    def set_env_config(self, motor_control_mode ="CPG", 
                       task_env                 ="FWD_LOCOMOTION", 
                       observation_space_mode   ="LR_COURSE_OBS",
                       terrain                  = None,
                       test_flagrun             = False,
                       add_noise                = True,
                       EPISODE_LENGTH           = 10,
                       render                   = False,
                       ):
        """
        Set environment configuration.

        Parameters:
        - motor_control_mode (str): Motor control mode (e.g., "CPG").
        - task_env (str): Task environment (e.g., "FWD_LOCOMOTION").
        - observation_space_mode (str): Observation space mode.
        """
        self.env_configs = {
            "motor_control_mode":     motor_control_mode,
            "task_env":               task_env,
            "observation_space_mode": observation_space_mode,
            "terrain":                terrain,
            "add_noise":              add_noise,
            'test_flagrun':           test_flagrun,
            "EPISODE_LENGTH":         EPISODE_LENGTH,
            "render":                 render
        }


    def initialize_environment(self):
        """
        Initialize the training environment and setup paths.
        """
        # Directory to save models and logs
        # SAVE_PATH = './logs/intermediate_models/'+ datetime.now().strftime("%m%d%y%H%M%S") + '/'
        self.SAVE_PATH = os.path.join('./logs/intermediate_models/', datetime.now().strftime("%m%d%y%H%M%S") + '/')   #SAVE_PATH
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        self.checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=self.SAVE_PATH, name_prefix='rl_model', verbose=2)  #FF00FF 定义保存频率 Define save frequency

        # Create environment
        env = lambda: QuadrupedGymEnv(**self.env_configs)
        self.env = make_vec_env(env, monitor_dir=self.SAVE_PATH, n_envs=self.num_envs)
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=False, clip_obs=100.)

    def load_pretrained_model(self):
        """
        Load a pre-trained model and its environment.
        """
        stats_path = os.path.join(self.log_dir, "vec_normalize.pkl")
        model_name = get_latest_model(self.log_dir)

        # Re-create environment
        env = lambda: QuadrupedGymEnv(**self.env_configs)
        self.env = make_vec_env(env, monitor_dir=self.SAVE_PATH, n_envs=self.num_envs, vec_env_cls=SubprocVecEnv)
        self.env = VecNormalize.load(stats_path, self.env)

        if self.algorithm == "PPO":
            self.model = PPO.load(model_name, self.env)
        elif self.algorithm == "SAC":
            self.model = SAC.load(model_name, self.env)
        print("\nLoaded model:", model_name, "\n")

    def create_model(self):
        """
        Create a new model for training.
        """
        policy_kwargs = dict(net_arch=[256, 256])
        if self.algorithm == "PPO":
            ppo_config = {
                "gamma": 0.99,
                "n_steps": int(4096 / self.num_envs),
                "ent_coef": 0.0,
                "learning_rate": lambda f: 1e-4,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "gae_lambda": 0.95,
                "batch_size": 128,
                "n_epochs": 10,
                "clip_range": 0.2,
                "clip_range_vf": 1, 
                "verbose": 1,                  # 是否打开日志 # open log or not #00FFFF 1 means open, 0 means close
                "policy_kwargs": policy_kwargs,
                "device": self.gpu_arg
            }
            self.model = PPO('MlpPolicy', self.env, **ppo_config)
        elif self.algorithm == "SAC":
            sac_config = {
                "learning_rate": 1e-4,
                "buffer_size": 300000,
                "batch_size": 256,
                "ent_coef": 'auto',
                "gamma": 0.99,
                "tau": 0.005,
                "train_freq": 1,
                "gradient_steps": 1,
                "learning_starts": 10000,
                "verbose": 1,                  # 是否打开日志 # open log or not #00FFFF 1 means open, 0 means close
                "policy_kwargs": policy_kwargs,
                "device": self.gpu_arg
            }
            self.model = SAC('MlpPolicy', self.env, **sac_config)
        else:
            raise ValueError(self.algorithm + " not implemented")

    def train(self, total_timesteps=1000000):
        """
        Train the model.

        Parameters:
        - total_timesteps (int): Total timesteps for training.
        """
        self.initialize_environment()

        if self.load_last_train:
            self.load_pretrained_model()
        else:
            self.create_model()

        # Train the model
        self.model.learn(total_timesteps=total_timesteps, log_interval=1, callback=self.checkpoint_callback)



    def save_model(self):
        """
        Save the trained model and environment statistics.
        """
        self.model.save(os.path.join(self.SAVE_PATH, "rl_model"))
        self.env.save(os.path.join(self.SAVE_PATH, "vec_normalize.pkl"))
        if self.algorithm == "SAC":
            self.model.save_replay_buffer(os.path.join(self.SAVE_PATH, "off_policy_replay_buffer"))


# 示例使用
if __name__ == "__main__":
    trainer = QuadrupedTrainer(algorithm="SAC", num_envs=1, load_last_train=False)
    trainer.set_env_config(motor_control_mode     ="CPG", 
                           task_env               ="FWD_LOCOMOTION", 
                           observation_space_mode ="LR_COURSE_OBS",
                           terrain                = None,
                           test_flagrun           = False,
                           add_noise              = True)
    trainer.train(total_timesteps=1000000)
    trainer.save_model()
