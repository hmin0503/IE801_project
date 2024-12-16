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
import gymnasium as gym
import torch
import config_transfer as config
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
    wandb.init(project=config.project_train,name=config.name_train, notes=config.notes, sync_tensorboard=config.sync_tensorboard)
    #wandb_callback is a callback that logs the training process to wandb, this is done because wandb.watch() does not work with sb3
    wandb_callback = WandbCallback()


'''
Set up the environment
'''
# Create multiple environments and wrap them correctly
env = make_atari_env(config.env_id, n_envs=config.n_envs, seed=config.seed)
env = VecFrameStack(env, n_stack=config.n_stack)
print(f"set up the environment - {config.env_id}")

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
print(f"set up the model - {model}")


if config.transfer_type != False:
    source_model = A2C.load(
            config.pretrained_model_path,
            verbose=config.verbose,
            tensorboard_log=config.log_dir,
            custom_objects={"lr_schedule": lambda _: config.learning_rate}
        )
    print(f"successfully load pretrained model.")

# if config.transfer_type == "all":
#     print("all networks transfered.")
#     model.policy.pi_features_extractor.load_state_dict(source_model.policy.features_extractor.state_dict())
#     model.policy.vf_features_extractor.load_state_dict(source_model.policy.vf_features_extractor.state_dict())
#     model.policy.features_extractor.load_state_dict(source_model.policy.features_extractor.state_dict())

# elif config.transfer_type == "policy_network_transfer":
#     print("policy network transfered.")
#     model.policy.pi_features_extractor.load_state_dict(source_model.policy.pi_features_extractor.state_dict())

# elif config.transfer_type == "value_network_transfer":
#     print("value network transfered.")
#     model.policy.vf_features_extractor.load_state_dict(source_model.policy.vf_features_extractor.state_dict())

# elif config.transfer_type == "shared_feature_extractor":
#     print("shared feature extractor transfered.")
#     model.policy.features_extractor.load_state_dict(source_model.policy.features_extractor.state_dict())
#     # config.policy_kwargs = {"features_extractor": source_model.policy.features_extractor}

if config.transfer_type == "all":
    print("all networks transfered.")
    model.policy.value_net.load_state_dict(source_model.policy.value_net.state_dict())
    model.policy.policy.load_state_dict(source_model.policy.policy.state_dict(), strict = False)
    model.policy.features_extractor.load_state_dict(source_model.policy.features_extractor.state_dict())

elif config.transfer_type == "policy_network_transfer":
    print("policy network transfered.")
    model.policy.load_state_dict(source_model.policy.state_dict(), strict = False)

elif config.transfer_type == "value_network_transfer":
    print("value network transfered.")
    model.policy.value_net.load_state_dict(source_model.policy.value_net.state_dict(), strict = False)

elif config.transfer_type == "policy_value_network_transfer":
    print("policy and value network transfered.")
    # model.policy.load_state_dict(source_model.policy.state_dict(), strict = False)
    model.policy.features_extractor.load_state_dict(source_model.policy.features_extractor.state_dict())
    model.policy.value_net.load_state_dict(source_model.policy.value_net.state_dict(), strict = False)


#Load the model if config.pretrained is set to True in config.py
if config.pretrained:
    # Load the pretrained model without an environment first
    model = A2C.load(
        config.pretrained_model_path,
        verbose=config.verbose,
        tensorboard_log=config.log_dir,
        custom_objects={"lr_schedule": lambda _: config.learning_rate}
    )

    if model.action_space != env.action_space:
        print(f"Model's action space: {model.action_space}")
        print(f"Environment's action space: {env.action_space}")
        
        # Modify the policy's action network to match the new action space
        model.policy.action_net = torch.nn.Linear(model.policy.features_dim, env.action_space.n)
        model.action_space = env.action_space

    unzip_file(config.pretrained_model_path, config.unzip_pretrained_path) 
    model.policy.load_state_dict(torch.load(os.path.join(config.unzip_pretrained_path, "policy.pth"), weights_only=True), strict=False)
    model.policy.optimizer.load_state_dict(torch.load(os.path.join(config.unzip_pretrained_path, "policy.optimizer.pth"), weights_only=True))
    
    # Check if the action spaces are different and modify the action network
        
    # Set the environment for the loaded model
    model.set_env(env)

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
    model.learn(total_timesteps=config.total_timesteps, log_interval=config.log_interval, callback=[checkpoint_callback, wandb_callback, custom_callback], progress_bar=True)
else:
    model.learn(total_timesteps=config.total_timesteps, log_interval=config.log_interval, callback=[checkpoint_callback, custom_callback], progress_bar=True)
#Save the model 
model.save(config.saved_model_path[:-4]) #remove the .zip extension from the path


''' 
Close the environment and finish the logging
'''
env.close()
if config.log_to_wandb:
    wandb.finish()