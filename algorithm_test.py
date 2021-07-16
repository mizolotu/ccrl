from gym.envs.classic_control import PendulumEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.ppo import PPO as ppo
from stable_baselines.ppo.policies import MlpPolicy
from stable_baselines.common.utils import set_random_seed

def make_env(env_class, seed):
    fn = lambda: env_class(seed)
    return fn

if __name__ == '__main__':

    env_class = PendulumEnv
    nenvs = 4
    timesteps = nenvs * int(1e6)
    set_random_seed(seed=0)

    env_fns = [make_env(env_class, seed) for seed in range(nenvs)]
    env = SubprocVecEnv(env_fns)

    model = ppo(MlpPolicy, env, n_steps=64, batch_size=64, ent_coef=0.01, tensorboard_log='log', verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=1)