from stable_baselines3.common.utils import get_latest_run_id
import torch
    

'''FILE TO STORE ALL THE CONFIGURATION VARIABLES'''
#environment_id
env_id = "BreakoutNoFrameskip-v4" # "PongNoFrameskip-v4"

#pretrained is a boolean that indicates if a pretrained model will be loaded
pretrained = True # Set to True if you want to load a pretrained model
transfer_type = False # "all" # "all", "policy_network_transfer", "value_network_transfer",  "shared_feature_extractor"

#check_freq is the frequency at which the callback is called, in this case, the callback is called every 2000 timesteps
check_freq = 2000

#save_path is the path where the best model will be saved
save_path = f'./DQN_{env_id}_{transfer_type}_1M_save_path'
checkpoint_path = f"./DQN_{env_id}_{transfer_type}_checkpoint"
 
#log_dir is the path where the logs will be saved
log_dir = f'./log_dir_{env_id}'

'''
Saved model path
'''
saved_model_path = f"./DQN_{env_id}_{transfer_type}.zip"
unzip_file_path = f"./DQN_{env_id}_{transfer_type}_unzipped"

'''
Hyperparameters of the model {policy, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, target_update_interval, exploration_fraction, exploration_initial_eps, exploration_final_eps, max_grad_norm, stats_window_size, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model, total_timesteps, log_interval}
'''
#policy is the policy of the model, in this case, the model will use a convolutional neural network
policy = "CnnPolicy"

#verbose is the verbosity level: 0 no output, 1 info, 2 debug
verbose=1 # 2

#seed is the seed for the pseudo random number generator
seed=0 # None

#device is the device on which the model will be trained
device= "cuda" if torch.cuda.is_available() else "cpu"

#_init_setup_model is a boolean that indicates if the model will be initialized
_init_setup_model=True

'''
Environment variables
'''
#n_stack is the number of frames stacked together to form the input to the model
n_stack = 4


'''
Wandb configuration
'''
#log_to_wandb is a boolean that indicates if the training process will be logged to wandb
log_to_wandb = True

# project is the name of the project in wandb
project_train = f"{env_id}-DQN-train"
project_test = f"{env_id}-DQN-test"

#entity is the name of the team in wandb

#name is the name of the run in wandb
name_train = f"DQN_{env_id}_train_{transfer_type}"
name_test = f"DQN_{env_id}_test_{transfer_type}"
#notes is a description of the run
notes = f"DQN_{env_id} with parameters: {locals()}" #locals() returns a dictionary with all the local variables, in this case, all the variables in this file
#sync_tensorboard is a boolean that indicates if the tensorboard logs will be synced to wandb
sync_tensorboard = True


'''
Test configuration
'''
test_episodes = 100