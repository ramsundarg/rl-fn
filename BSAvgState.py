import math
import gym
from gym import spaces
import numpy as np
import importlib
import tensorflow as tf

from mlflow import log_metric, log_param, log_artifacts



class BSAvgState(gym.Env):
    """
    Custom discrete-time Black-Scholes environment with one risky-asset and bank account
    The environment simulates the evolution of the investor's portfolio according to
    a discrete version of the wealth SDE.

    At the end of the investment horizon the reward is equal to U(V(T)), else zero.
    """
    def power_utility(self,x):
        return tf.pow(x, self.env['b']) / self.env['b']

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
        self.env = env
        self.mu = env['mu']
        self.sigma = env['sigma']
        self.r = env['r']
        self.T = env['T']
        self.dt = env['dt']
        self.V_0 = env['V_0']
        self.m = 100
        uf = env.get('U_2','math.log')
        if uf == 'np.log':
            self.U_2 = np.log
        else:
            self.U_2 = self.power_utility
        
        
        #self.U_2 = env.get('U_2',math.log)
        self.reset()
        
        self.action_space = spaces.Box(low=-1, high=1,
                               shape=(1,), dtype=np.float32)

        # Observations: t in [0,T]; V_t in [0, infinity)
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([self.T, float("inf")]),
                                            shape=(2,),
                                            dtype=np.float32)
 
        # Action space (denotes fraction of wealth invested in risky asset, excluding short sales)

    def wealth_update(self,r, mu, sigma, dt,  action, dW_t):
        return np.exp(
            (r + action[0] * (mu - r) - 0.5 * (action[0] ** 2) * (sigma ** 2)) * dt
            + action[0] * sigma * dW_t
        )
    def step(self, action):
        """Execute one time step within the environment

        :params action (float): investment in risky asset
        """
        # Update Wealth (see wealth dynamics, Inv. Strategies script (by Prof. Zagst) Theorem 2.18):
        dW_t = np.random.normal(loc=0, scale=math.sqrt(self.dt),size=1)
        # Wealth process update via simulation of the exponent
        V_t = self.V_t * self.wealth_update(self.r, self.mu, self.sigma, self.dt, action, dW_t)
        self.t += self.dt

        done = self.t >= self.T
        reward=0
        if done:
            reward = self.U_2(V_t)
        self.V_t = np.mean(V_t)
        # Additional info (not used for now)
        info = {}

        return self._get_obs(), np.mean(reward), done, info


    def _get_obs(self):
        return np.array([self.t, self.V_t], dtype=np.float32)

    def reset(self):
        """Reset the state of the environment to an initial state"""
        self.t = 0
        if isinstance(self.V_0, str):
            self.V_t = eval(self.V_0)
        else:
            self.V_t = self.V_0

        return self._get_obs()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        raise NotImplementedError("Use the discrete_bs_render environment for rendering")