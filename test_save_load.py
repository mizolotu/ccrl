import os, shutil
import os.path as osp

from gym.envs.box2d import LunarLander, LunarLanderContinuous
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.ppo import PPO as ppo
from stable_baselines.ppo.policies import MlpPolicy
from stable_baselines.common.utils import set_random_seed

from config import *

def make_env(env_class, seed):
    #fn = lambda: env_class(seed)
    fn = lambda: env_class()
    return fn

if __name__ == '__main__':

    # params

    env_classes = [LunarLanderContinuous, LunarLander]
    nenvs = 4
    timesteps = nenvs * int(1e4)
    set_random_seed(seed=0)
    alg = ppo

    # clean tensorboard test dir

    test_tensorboard_log_dir = 'test'
    if osp.isdir(TENSORBOARD_DIR):
        for subdir in os.listdir(TENSORBOARD_DIR):
            dpath = osp.join(TENSORBOARD_DIR, subdir)
            if osp.isdir(dpath) and subdir.startswith(test_tensorboard_log_dir):
                shutil.rmtree(dpath)

    for env_class in env_classes:

        # create env

        env_fns = [make_env(env_class, seed) for seed in range(nenvs)]
        env = SubprocVecEnv(env_fns)

        # create and train model

        model = ppo(MlpPolicy, env, n_steps=64, batch_size=64, ent_coef=0.01, tensorboard_log=TENSORBOARD_DIR, verbose=1, policy_kwargs=dict(net_arch = [256, dict(pi=[256], vf=[256])]))
        model.learn(total_timesteps=timesteps, log_interval=1, tb_log_name=test_tensorboard_log_dir)

        # save and delete model

        modeldir = osp.join(MODEL_DIR, env_class.__name__, alg.__name__)
        model.save(modeldir)
        del model

        # load model

        model = ppo(MlpPolicy, env, tensorboard_log=TENSORBOARD_DIR, loadpath=modeldir)
        model.learn(total_timesteps=timesteps, log_interval=1, tb_log_name=test_tensorboard_log_dir)