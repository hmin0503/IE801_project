'''
Code for training the A2C agent on the Breakout environment using stable baselines 3's A2C implementation

Note: For understanding the code, it is recommended to read the documentation of stable baselines 3's A2C implementation and to
understand the parameters of the model, go to config.py and read the comments of each parameter.
'''

'''
Import the necessary libraries and modules
'''
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import config
import wandb
from wandb.integration.sb3 import WandbCallback
from utils import make_env, unzip_file, CustomWandbCallback
import os
import tensorboard as tb
from tqdm import tqdm


'''
Set up the appropriate directories for logging and saving the model
'''
os.makedirs(config.log_dir, exist_ok=True)
os.makedirs(config.save_path, exist_ok=True)

#Create the callback that logs the mean reward of the last 100 episodes to wandb
custom_callback = CustomWandbCallback(config.check_freq, config.save_path)
checkpoint_callback = CheckpointCallback(save_freq=config.check_freq, save_path=config.checkpoint_path)

'''
Set up loging to wandb
'''
#Set wandb to log the training process
if config.log_to_wandb:
    wandb.init(project=config.project_train, name=config.name_train, notes=config.notes, sync_tensorboard=config.sync_tensorboard)
    #wandb_callback is a callback that logs the training process to wandb, this is done because wandb.watch() does not work with sb3
    wandb_callback = WandbCallback()


'''
Set up the environment
'''
# Create multiple environments and wrap them correctly
env = make_atari_env(config.env_id, n_envs=config.n_envs, seed=config.seed) #
env = VecFrameStack(env, n_stack=config.n_stack)

'''
CUSTOM CNN POLICY
'''
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        n_flatten = self.cnn(
            torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(64, features_dim))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear(self.cnn(observations)))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)


'''
Set up the model
'''
#Create the model with the parameters specified in config.py, go to config.py to see the meaning of each parameter in detail
model = A2C(
            policy = config.policy
            , env = env
            , verbose=config.verbose
            , tensorboard_log=config.log_dir
            , seed=config.seed
            , policy_kwargs=config.policy_kwargs
            # , sde_sample_freq=config.sde_sample_freq 
            , normalize_advantage=config.normalize_advantage
            # , stats_window_size=config.stats_window_size
            # , _init_setup_model=config._init_setup_model
            , device=config.device
            , learning_rate=config.learning_rate
            , n_steps=config.n_steps
            , gamma=config.gamma
            # , gae_lambda=config.gae_lambda
            , ent_coef=config.ent_coef
            , vf_coef=config.vf_coef
            # , max_grad_norm=config.max_grad_norm
            , rms_prop_eps=config.rms_prop_eps
            , use_rms_prop=config.use_rms_prop
            # , use_sde=config.use_sde,
            )

#Load the model if config.pretrained is set to True in config.py
if config.pretrained:
    model = A2C.load(config.saved_model_path, env=env, verbose=config.verbose, tensorboard_log=config.log_dir)
    #Unzip the file a2c_Breakout_1M.zip and store the unzipped files in the folder a2c_Breakout_unzipped
    unzip_file(config.saved_model_path, config.unzip_file_path) 
    model.policy.load_state_dict(torch.load(os.path.join(config.unzip_file_path, "policy.pth")))
    model.policy.optimizer.load_state_dict(torch.load(os.path.join(config.unzip_file_path, "policy.optimizer.pth")))

'''
Train the model and save it
'''
#model.learn will train the model for 1e6 timesteps, timestep is the number of actions taken by the agent, 
# in a game like breakout, the agent takes an action every frame, then the number of timesteps is the number of frames,
# which is the number of frames in 1 game multiplied by the number of games played.
#The average number of frames in 1 game is 1000, so 1e6 timesteps is 1000 games more or less.
#log_interval is the number of timesteps between each log, in this case, the training process will be logged every 100 timesteps.
#callback is a callback that logs the training process to wandb, this is done because wandb.watch() does not work with sb3

if config.log_to_wandb:
    model.learn(total_timesteps=config.total_timesteps, log_interval=config.log_interval, callback=[wandb_callback, custom_callback, checkpoint_callback], progress_bar=True)
else:
    model.learn(total_timesteps=config.total_timesteps, log_interval=config.log_interval, callback=[custom_callback, checkpoint_callback] , progress_bar=True)
#Save the model 
model.save(config.saved_model_path[:-4]) #remove the .zip extension from the path


''' 
Close the environment and finish the logging
'''
env.close()
if config.log_to_wandb:
    wandb.finish()