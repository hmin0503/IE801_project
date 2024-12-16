from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

def render_model(env, filename):
    loaded_model = A2C.load(filename)
    obs = env.reset()
    for i in range(10000):
        action, _states = loaded_model.predict(obs, deterministic=True) 
        obs, rewards, dones, info = env.step(action) 
        env.render("human")


if __name__ == "__main__":

    n_envs = 4
    env_id = 'Breakout-v0'
    env = make_atari_env(env_id, n_envs=n_envs, seed=0)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    filename = "checkpoints//rl_model_600000_steps.zip"
    render_model(env, filename)