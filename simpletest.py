from gym.envs.classic_control.pendulum import PendulumEnv
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

from common.vec_env.subproc_vec_env import SubprocVecEnv
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.policies import MlpPolicy, MlpLstmPolicy, MemoryPolicy

from a2c.a2c import A2C
from ppo2.ppo2 import PPO2
from acktr.acktr import ACKTR

def make_env():
    fn = lambda: PendulumEnv()
    #fn = lambda: Continuous_MountainCarEnv()
    return fn

if __name__ == '__main__':

    nenvs = 16
    env_fns = [make_env() for _ in range(nenvs)]
    env = SubprocVecEnv(env_fns)

    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor

    #model = A2C(MlpPolicy, env, verbose=1)
    #model = A2C(MemoryPolicy, env, verbose=1)
    model = PPO2(MemoryPolicy, env, verbose=1, nminibatches=nenvs)
    #model = ACKTR(LstmPolicy, env, verbose=1)

    model.learn(total_timesteps=nenvs*1000000)