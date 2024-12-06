import copy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3 import A2C
import gym
from stable_baselines3 import PPO
# from ale_py import ALEInterface
import torch
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper 
from stable_baselines3.common.callbacks import BaseCallback


def main():
    # There already exists an environment generator
    # that will make and wrap atari environments correctly.
    # Here we are also multi-worker training (n_envs=4 => 4 environments)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_envs = 1
    env_id = 'Pong-v0'
    env = make_atari_env(env_id, n_envs=n_envs, seed=0)
    
    
    # Frame-stacking with 4 frames
    train_env = VecFrameStack(env, n_stack=1)
    train_env = VecTransposeImage(train_env)

    video_folder = "logs/videos_training/"
    video_length = 1000
    
    train_env = VecVideoRecorder(
        train_env,
        video_folder,
        record_video_trigger=lambda x: x % 10000 == 0,  # Record every 10000 timesteps
        video_length=video_length,
        name_prefix=f"training-agent-{env_id}"
    )

    model = A2C(
        'CnnPolicy', 
        train_env, 
        verbose=1,
        learning_rate=0.0007,
        n_steps=10 * n_envs,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        tensorboard_log="./a2c_atari_tensorboard/",
        device = device
    )

    env = make_atari_env(env_id, n_envs=1, seed=0)

    eval_env = VecFrameStack(env, n_stack=1)
    eval_env = VecTransposeImage(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./checkpoints/")
    
 
    model.learn(
        total_timesteps=10000*100,
        callback=[eval_callback, checkpoint_callback]
        )

    # Save the model
    model.save("a2c_atari_breakout")
    loaded_model = A2C.load("a2c_atari_breakout")

   
    video_folder = "logs/videos/"
    video_length = 100

    vec_env = model.get_env()

    

    # Record the video starting at the first step
    vec_env = VecVideoRecorder(vec_env, video_folder,
                        record_video_trigger=lambda x: x == 0, video_length=video_length,
                        name_prefix=f"trained-agent-{env_id}")
    obs = vec_env.reset()
    
    for _ in range(video_length + 1):
       action, _states = loaded_model.predict(obs, deterministic=True) 
       obs, rewards, dones, info = vec_env.step(action) 
       vec_env.render("human")
    vec_env.close()


if __name__ == "__main__":
    main()




