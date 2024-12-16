import os
from stable_baselines3.common.utils import get_latest_run_id
import torch

'''FILE TO STORE ALL THE CONFIGURATION VARIABLES'''
#environment_id
env_id = "BreakoutNoFrameskip-v4" # "PongNoFrameskip-v4"

#pretrained is a boolean that indicates if a pretrained model will be loaded
pretrained = False # Set to True if you want to load a pretrained model
transfer_type = False # "all" # "all", "policy_network_transfer", "value_network_transfer",  "shared_feature_extractor"

'''
Pretrained model path
'''
pretrained_model_path = ""
unzip_pretrained_path = ""

#check_freq is the frequency at which the callback is called, in this case, the callback is called every 2000 timesteps
check_freq = 2000

#save_path is the path where the best model will be saved
save_path = f'./a2c_{env_id}_1M_save_path'
checkpoint_path = f"./a2c_{env_id}_checkpoint"

#log_dir is the path where the logs will be saved
log_dir = f'./log_dir_{env_id}'

'''
Saved model path
'''
saved_model_path = f"./a2c_model_{env_id}.zip"
unzip_file_path = f"./a2c_model_{env_id}_unzipped"


'''
Hyperparameters of the model {learning_rate, gamma, device, n_steps, gae_lambda, ent_coef, vf_coef, max_grad_norm, rms_prop_eps, use_rms_prop, use_sde, sde_sample_freq, normalize_advantage}
'''
#policy is the policy of the model, in this case, the model will use a convolutional neural network
policy = "CnnPolicy"

#learning_rate is the learning rate of the model
learning_rate = 5e-4   #first trial: 5e-4   #second trial: 1e-4  #third trial: 1e-3  #fourth trial: 5e-5 #fifth trial: 5e-5 gamma = 0.90 #sixth trial: 1e-4 gamma = 0.90 #seventh trial: 5e-4 gamma = 0.90

#gamma is the discount factor
gamma = 0.99

#device is the device where the model will be trained, if cuda is available, the model will be trained in the gpu, otherwise, it will be trained in the cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

#n_steps is the number of steps taken by the model before updating the parameters
n_steps = 24

#gae_lambda is the lambda parameter of the generalized advantage estimation, set to 1 to disable it
gae_lambda = 1

#ent_coef is the entropy coefficient, set to 0 to disable it
ent_coef = 0.01

#vf_coef is the value function coefficient, If we set it to 0.5, then the value function loss will be half the policy loss
vf_coef = 0.25

#max_grad_norm is the maximum value for the gradient clipping
max_grad_norm = 0.5

#rms_prop_eps is the RMSProp optimizer epsilon parameter, which is a small value added to the denominator to avoid division by zero
rms_prop_eps = 0.00001

#use_rms_prop is a boolean that indicates if the RMSProp optimizer will be used
use_rms_prop = True

#use_sde is a boolean that indicates if the stochastic differential equation will be used
#The stochastic differential equation is a method to add noise to the actions taken by the agent to improve exploration
use_sde = False

#sde_sample_freq is the frequency at which the noise is added to the actions. If set to -1, the noise will be added every timestep
sde_sample_freq = -1

#normalize_advantage is a boolean that indicates if the advantage will be normalized, by normalizing the advantage, 
# the variance of the advantage is reduced, this is done to improve the training process because the advantage is used to calculate the policy loss
normalize_advantage = False

#stats_window_size is the size of the window used to calculate the mean and standard deviation of the advantage
stats_window_size = 100

#tensorboard_log is the path where the tensorboard logs will be saved, in our case, the logs will be saved in the log_dir
tensorboard_log = log_dir

#policy_kwargs is a dictionary with the keyword arguments for the policy. If None, it will use the default arguments
policy_kwargs = None

#verbose is the verbosity level: 0 no output, 1 info, 2 debug
verbose = 1

#seed is the seed for the pseudo random number generator used by the model. It is set to None to use a random seed,
# and set to 0 to use a fixed seed for reproducibility
seed = 0 # None

#_init_setup_model is a boolean that indicates if the model will be initialized after being created, set to True to initialize the model
_init_setup_model = True

#total_timesteps is the total number of timesteps that the model will be trained. In this case, the model will be trained for 1e7 timesteps
#Take into account that the number of timesteps is not the number of episodes, in a game like breakout, the agent takes an action every frame,
# then the number of timesteps is the number of frames, which is the number of frames in 1 game multiplied by the number of games played.
#The average number of frames in 1 game is 1000, so 1e7 timesteps is 1000 games more or less.
total_timesteps = int(1e7)

#log_interval is the number of timesteps between each log, in this case, the training process will be logged every 100 timesteps.
log_interval = 100

'''
Environment variables
'''
#n_stack is the number of frames stacked together to form the input to the model
n_stack = 4
#n_envs is the number of environments that will be run in parallel
n_envs = 16


'''
Wandb configuration
'''
#log_to_wandb is a boolean that indicates if the training process will be logged to wandb
log_to_wandb = True

#project is the name of the project in wandb
project_train = f"{env_id}-a2c-train"
project_test = f"{env_id}-a2c-test"

#entity is the name of the team in wandb

#name is the name of the run in wandb
name_train = f"a2c_{env_id}_train"
name_test = f"a2c_{env_id}_test"
#notes is a description of the run
notes = f"a2c_{env_id} with parameters: {locals()}" #locals() returns a dictionary with all the local variables, in this case, all the variables in this file
#sync_tensorboard is a boolean that indicates if the tensorboard logs will be synced to wandb
sync_tensorboard = True

'''
Test configuration
'''
test_episodes = 100