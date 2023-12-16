from typing import Tuple

import numpy as np
import gym
from gym import spaces
import control as ct
from control.matlab import *


class AnestheisaEnv(gym.Env):
    def __init__(self, age, weight, height, lbm, t_s):
        super(AnestheisaEnv, self).__init__()
        self.age = age
        self.weight = weight
        self.height = height
        self.lbm = lbm
        self.t_s = t_s
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

        # Define the continuous action space
        self.action_space = spaces.Box(low=-0, high=2, shape=(1,), dtype=np.float32)

        # Initialize the state
        self.state = 100  # Example: Random initial state between 0 and 1

    def pk_model(self):
        v1p = 4.27
        v2p = 18.9 - 0.391 * (self.age - 53)
        v3p = 238     # [l]
        cl1p = 1.89 + 0.0456 * (self.weight - 77) - 0.0681 * (self.lbm - 59) + 0.0264 * (self.height - 177)
        cl2p = 1.29 - 0.024 * (self.age - 53)
        cl3p = 0.836  # [l.min ^ (-1)]

        clearance = np.array([cl1p, cl2p, cl3p])
        volume = np.array([v1p, v2p, v3p])
        n = len(clearance)
        k1 = clearance / volume[0]
        k2 = clearance[1:] / volume[1:]
        a = np.vstack((np.hstack((-np.sum(k1), k1[1:])), np.hstack((np.transpose(k2)[:, None], -np.diag(k2)))))
        b = np.vstack(([1 / volume[0]], np.zeros((n - 1, 1))))
        c = np.array([[1, 0, 0]])
        d = np.array([[0]])
        # a = a / 60
        pk_sys = ct.ss(a, b, c, d)
        return pk_sys

    def pd_linear_model(self):
        ke0 = 0.456    # [min^(-1)]
        t_d = 20
        num = np.array([ke0])
        den = np.array([1, ke0])
        sys = ct.tf(num, den)
        time_delay_pad_app = ct.tf(ct.pade(t_d)[0], ct.pade(t_d)[1])
        pd_lin_sys = ct.series(sys, time_delay_pad_app)
        # pd_lin_sys = sys
        return pd_lin_sys

    def pd_model_hillfunction(self, ce):
        e0 = 100
        gamma = 2
        ce50 = 4.16
        if ce < 0:
            ce = 0
        # ce[ce<0] = 0
        e = e0 - e0*ce**gamma/(ce**gamma + ce50**gamma)
        return e

    def step(self, action):
        # Clip the action to be within the action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Update the state based on the action
        pk_pd_lin = ct.series(self.pk_model(), self.pd_linear_model())
        # print(pk_pd_lin)
        yout, tout, xout = lsim(pk_pd_lin,
                                U=action*np.ones(self.t_s+1),
                                T=np.linspace(0, self.t_s, self.t_s+1))
        # print(yout)
        # self.state[0:3] = np.array(yout)
        self.state = np.array(self.pd_model_hillfunction(yout[-1]))
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        # Define the reward function (example: reward is higher when state is closer to 1)
        # reward = np.exp(-abs(self.state - 1))
        reward = -(self.state-50)**2
        epsilon = 5
        # Check if the episode is done
        if np.abs(self.state-50) < epsilon:
            done = True
        else:
            done = False

        return self.state, reward, done

    def reset(self):
        # Reset the environment to a random initial state
        self.state = 100-np.random.rand(1)
        return self.state

    def render(self, mode='human'):
        # Implement rendering here if needed
        pass

    def close(self):
        # Implement any cleanup code here if needed
        pass


t_s = 900
age = 45
weight = 64
height = 171
lbm = 52
env = AnestheisaEnv(age, weight, height, lbm, t_s)
obs = env.reset()
# print(type(obs))
for _ in range(100):
    action = env.action_space.high
    # action = env.action_space.sample()  # Sample a random action
    obs, reward, done = env.step(action)
    print(f"State: {obs}, Reward: {reward}, Done: {done}, Action:{action}")

env.close()


