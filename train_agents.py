import os, shutil, sys
import os.path as osp

from baselines.common.vec_env import SubprocVecEnv
from baselines.ddpg.ddpg import learn as learn_ddpg
from baselines.a2c.a2c import learn as learn_a2c
from baselines.trpo_mpi.trpo_mpi import learn as learn_trpo
from baselines.ppo2.ppo2 import learn as learn_ppo
from baselines import logger

from gym.envs.classic_control.pendulum import PendulumEnv
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.envs.box2d.car_racing import CarRacing
from gym.envs.box2d.bipedal_walker import BipedalWalker
from gym.envs.box2d.lunar_lander import LunarLanderContinuous

def make_env(env_class, k=None):
    if k is not None:
        fn = lambda : env_class(stack=k)
    else:
        fn = lambda: env_class()
    return fn

def clean_dir(dir_name):
    for the_file in os.listdir(dir_name):
        file_path = osp.join(dir_name, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def test_alg_on_env(env_class, algorithm, network, ne, ns, tt, ld='log'):
    if 'cnn' in network or 'lstm' in network:
        k = 4
        env_fns = [make_env(env_class, k) for _ in range(ne)]
    else:
        env_fns = [make_env(env_class) for _ in range(ne)]
    train_envs = SubprocVecEnv(env_fns)
    logdir = '{0}/{1}/{2}_{3}/'.format(ld, env_class.__name__, algorithm['name'], network)
    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)
    if int(tt/(ne*ns*100)) < 1:
        log_interval = int(tt/(ne*ns*100))
    else:
        log_interval = 1
    algorithm['learn'](env=train_envs, network=network, nsteps=ns, total_timesteps=tt, log_interval=log_interval)

if __name__ == '__main__':

    if 'cpu' in sys.argv[1:]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    env_classes = [
        PendulumEnv,
        Continuous_MountainCarEnv,
        CarRacing,
        LunarLanderContinuous,
        BipedalWalker
    ]

    algorithms = [
        {'name': 'ddpg', 'learn': learn_ddpg},
        {'name': 'a2c', 'learn': learn_a2c},
        {'name': 'trpo', 'learn': learn_trpo},
        {'name': 'ppo', 'learn': learn_ppo},
    ]

    networks = [
        'mlp2_64',
        'mlp2_256',
        'mlp2_1024',
        'mlp3_64',
        'mlp3_256',
        'mlp3_1024',
        'cnn_mlp_64',
        'cnn_mlp_256',
        'cnn_mlp_1024',
        'cnn2_64',
        'cnn2_256',
        'cnn2_1024',
        'lstm1_64',
        'lstm1_256',
        'lstm2_64',
        'lstm2_256',
        'alstm_64',
        'alstm_256',
        'blstm_64',
        'blstm_256',
        'ablstm_64',
        'ablstm_256'
    ]

    n_envs = int(sys.argv[1])
    n_episodes = int(sys.argv[2])

    n_steps = 125
    total_timesteps = n_episodes * n_steps * n_envs
    print('Total time steps: {0}'.format(total_timesteps))

    for network in networks:
        if '' in network:
            print(network)
            test_alg_on_env(env_classes[0], algorithms[3], network, ne=n_envs, ns=n_steps, tt=total_timesteps)