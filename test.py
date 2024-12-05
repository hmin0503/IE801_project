import copy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3 import A2C
import gym
from stable_baselines3 import PPO
# from ale_py import ALEInterface
import torch



def main():
        # There already exists an environment generator
    # that will make and wrap atari environments correctly.
    # Here we are also multi-worker training (n_envs=4 => 4 environments)
    n_envs = 4
    env_id = 'Breakout-v0'
    env = make_atari_env(env_id, n_envs=n_envs, seed=0)

    # Frame-stacking with 4 frames
    train_env = VecFrameStack(env, n_stack=4)
    train_env = VecTransposeImage(train_env)

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
    )

    env = make_atari_env(env_id, n_envs=1, seed=0)
    eval_env = VecFrameStack(env, n_stack=4)
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
    vec_env = model.get_env() 
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = loaded_model.predict(obs, deterministic=True) 
        obs, rewards, dones, info = vec_env.step(action) 
        vec_env.render("human")

if __name__ == "__main__":
    main()