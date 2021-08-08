import os, shutil
import os.path as osp

from gym.envs.box2d import LunarLander, LunarLanderContinuous
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.ppo import PPO as ppo
from stable_baselines.ppo.policies import MlpPolicy
from stable_baselines.common.utils import set_random_seed

from config import *

def make_env(env_class):
    fn = lambda: env_class()
    return fn

if __name__ == '__main__':

    # params

    env_classes = [LunarLanderContinuous, LunarLander]
    nenvs = 4
    timesteps = nenvs * int(1e5)
    set_random_seed(seed=0)
    alg = ppo

    # clean tensorboard test dir

    test_tensorboard_log_dir = 'test_save_load'
    if osp.isdir(TENSORBOARD_DIR):
        for subdir in os.listdir(TENSORBOARD_DIR):
            dpath = osp.join(TENSORBOARD_DIR, subdir)
            if osp.isdir(dpath) and subdir.startswith(test_tensorboard_log_dir):
                shutil.rmtree(dpath)

    for env_class in env_classes:

        # create env

        env_fns = [make_env(env_class) for _ in range(nenvs)]
        env = SubprocVecEnv(env_fns)

        # clean model dir

        modeldir = osp.join(MODEL_DIR, env_class.__name__, alg.__name__, MlpPolicy.__name__)
        if osp.isdir(modeldir):
            shutil.rmtree(modeldir)

        # clean progress dir

        logdir = osp.join(PROGRESS_DIR, env_class.__name__, alg.__name__, MlpPolicy.__name__)
        if osp.isdir(logdir):
            shutil.rmtree(logdir)

        # create and train model

        model = alg(MlpPolicy, env, n_steps=256, batch_size=256, tensorboard_log=TENSORBOARD_DIR, verbose=1, policy_kwargs=dict(net_arch = [256, dict(pi=[256], vf=[256])]), modelpath=modeldir, logpath=logdir)
        model.learn(total_timesteps=timesteps, log_interval=1, tb_log_name=test_tensorboard_log_dir)

        # delete model

        del model

        # load model and continue training

        model = alg(MlpPolicy, env, n_steps=256, batch_size=256, tensorboard_log=TENSORBOARD_DIR, verbose=1, policy_kwargs=dict(net_arch = [256, dict(pi=[256], vf=[256])]), modelpath=modeldir, logpath=logdir)
        model.learn(total_timesteps=timesteps, log_interval=1, tb_log_name=test_tensorboard_log_dir)