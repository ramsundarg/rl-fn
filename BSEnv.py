import math

import gym
import numpy as np
from gym import spaces
import importlib

from mlflow import log_metric, log_param, log_artifacts


def wealth_update(r, mu, sigma, dt,  action, dW_t):
    return np.exp(
        (r + action * (mu - r) - 0.5 * (action ** 2) * (sigma ** 2)) * dt
        + action * sigma * dW_t
    )

class BSEnv:
    """
    Custom discrete-time Black-Scholes environment with one risky-asset and bank account
    The environment simulates the evolution of the investor's portfolio according to
    a discrete version of the wealth SDE.

    At the end of the investment horizon the reward is equal to U(V(T)), else zero.
    """


    def __init__(self, env):
        """
        :params mu (float):         expected risky asset return
        :params sigma (float):      risky asset standard deviation
        :params r (float):          risk-less rate of return
        :params T (float):          investment horizon
        :params dt (float):         time-step size
        :params V_0 (float, tuple): initial wealth, if tuple (v_d, v_u) V_0 is uniformly drawn from [v_d, v_u]
        :params U_2 (callable):     utility function for terminal wealth
        """

        super().__init__()
        self.mu = env['mu']
        self.sigma = env['sigma']
        self.r = env['r']
        self.T = env['T']
        self.dt = env['dt']
        self.V_0 = env['V_0']
        p, m = env.get('U_2','math.log').rsplit('.', 1)
        mod = importlib.import_module(p)
        self.U_2 = getattr(mod, m)
        
        #self.U_2 = env.get('U_2',math.log)
        self.reset()

        # Action space (denotes fraction of wealth invested in risky asset, excluding short sales)
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(1,), dtype=np.float32)

        # Observations: t in [0,T]; V_t in [0, infinity)
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([self.T, float("inf")]),
                                            shape=(2,),
                                            dtype=np.float32)

    def step(self, action):
        """Execute one time step within the environment

        :params action (float): investment in risky asset
        """
        # Update Wealth (see wealth dynamics, Inv. Strategies script (by Prof. Zagst) Theorem 2.18):
        dW_t = np.random.normal(loc=0, scale=math.sqrt(self.dt))
        # Wealth process update via simulation of the exponent
        self.V_t *= wealth_update(self.r, self.mu, self.sigma, self.dt, action, dW_t)
        self.t += self.dt

        done = self.t >= self.T
        reward = 0
        if done:
            reward = self.U_2(self.V_t)

        # Additional info (not used for now)
        info = {}

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.array([self.t, self.V_t], dtype=np.float32)

    def reset(self):
        """Reset the state of the environment to an initial state"""
        self.t = 0
        if isinstance(self.V_0, tuple):
            self.V_t = 0.5 #np.random.uniform(low=0, high=1)
        else:
            self.V_t = 0.5

        return self._get_obs()

