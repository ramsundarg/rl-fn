import math
import gym
from gym import spaces
import numpy as np
import importlib
import tensorflow as tf

from mlflow import log_metric, log_param, log_artifacts
"""This module is for configuring an environment to be used for portfolio optimization"""


class BSAvgState(gym.Env):
    """
    Custom discrete-time Black-Scholes environment with one risky-asset and bank account
    The environment simulates the evolution of the investor's portfolio according to
    a discrete version of the wealth SDE.

    At the end of the investment horizon the reward is equal to U(V(T)), else zero.
    """
    def power_utility(self,x):
        return tf.pow(x*1.0, self.env['b']) / self.env['b']

    def __init__(self, env):
        """
        The initialization function. Note that 'env' is defined in the cfg file.


        Parameters
         mu (float):         expected risky asset return
         sigma (float):      risky asset standard deviation
         r (float):          risk-less rate of return
         T (float):          investment horizon
         dt (float):         time-step size
         V_0 (float, tuple): initial wealth, if tuple (v_d, v_u) V_0 is uniformly drawn from [v_d, v_u]
         U_2 (callable):     utility function for terminal wealth

        Returns 
         None
        """

        super().__init__()
        self.env = env
        self.mu = env['mu']
        self.sigma = env['sigma']
        self.r = env['r']
        self.T = env['T']
        self.dt = env['dt']
        self.V_0 = env['V_0']
        self.pre_load = False
        
        if env.get('load_variates',0)==1:
            self.rv = np.load('random_variates.npy')
            self.nexti = 0
            self.pre_load = True
        uf = env.get('U_2','np.log')
        if uf == 'np.log':
            self.U_2 = np.log
        else:
            self.U_2 = self.power_utility
        
        
        #self.U_2 = env.get('U_2',math.log)
        self.dP=None
 
        self.reset()
        
        self.action_space = spaces.Box(low=-1, high=1,
                               shape=(1,), dtype=np.float32)

        # Observations: t in [0,T]; V_t in [0, infinity)
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([self.T, float("inf")]),
                                            shape=(2,),
                                            dtype=np.float32)
        
        # Action space (denotes fraction of wealth invested in risky asset, excluding short sales)
    def wealth_grid_update(self,r, mu, sigma, dt,  action, dP):
        """
        Updates a complete mini batch of (state,action). dP is a vector of log returns, sampled from replay bufffer or could be generated from a distribution (See the Estimate version ). So in effect for each of the allocations in action, it takes all the returns specified in dP and computes a cross product of them.

        Parameters:
            r - riskfree rate
            mu - <not used currently , as these are implictly included in dP>
            sigma -  volatility of the risky asset
            dt - time step
            action - allocation in the risky asset
            dP - log shock retruns of the risky asset
        
        Returns:
            The updated wealth of each action across all the shock returns. If the size of the actions is n and size of shock returns is m, then the resulting wealth update will be of the dimension m * n

        """
        a = action
        return tf.exp((1-a)*r*dt + tf.matmul(action,tf.transpose(dP)) + 0.5*a*(1-a)*dt*(sigma**2))
    def wealth_update(self,r, mu, sigma, dt,  action, dP):
        """
        Wealth updates for (state,action). dP is a vector of log returns, sampled from replay bufffer or could be generated from a distribution (See the Estimate version ). So in effect for each of the allocations in action, it takes all the returns specified in dP and computes a cross product of them.

        Parameters:
            r - riskfree rate
            mu - <not used currently , as these are implictly included in dP>
            sigma -  volatility of the risky asset
            dt - time step
            action - allocation in the risky asset
            dP - log shock retruns of the risky asset
        
        Returns:
            The updated wealth of each action across all the shock returns. If the size of the actions is n and size of shock returns is m, then the resulting wealth update will be of the dimension m * n
        """

        a = action
        return tf.exp((1-a)*r*dt + action*dP + 0.5*a*(1-a)*dt*(sigma**2))

    def generate_random_returns(self,count):
            """
                Generates a random list of log returns to be used for the risky asset.

                Parameters:
                    count - Number of shock returns needed

                Returns:
                    Count number of Shock returns.

            """
            if self.pre_load:
                dW_t = np.array(np.roll(self.rv,self.nexti)[0:count]).reshape(-1)
                self.nexti = (self.nexti+count)%100000 # Max size of the random variates file
            else:
                dW_t = np.random.normal(loc=0, scale=math.sqrt(self.dt),size=(count))
            return (self.mu - 0.5*self.sigma**2)*self.dt + self.sigma*dW_t
    
    def generate_returns_given_variate(self,dW_t):
        """
        Not used currently. Deprecated
        """

        return (self.mu - 0.5*self.sigma**2)*self.dt + self.sigma*dW_t
    def peek_steps(self, state,action,count):
        """ 
            Computes the next state without actually entering into the state. It can transition into count number of states for any count >1.

            Parameters:
                (state,action) : The state,action tuple
                count - The number of new states to compute into

            Returns:
                The tuple (state(next state,next time),reward,isTerminalState,shock returns)            
        """
        # Update Wealth (see wealth dynamics, Inv. Strategies script (by Prof. Zagst) Theorem 2.18):
        dP = self.generate_random_returns(count)
        
        # Wealth process update via simulation of the exponent
        next_wealth = state[1] *self.wealth_update(self.r, self.mu, self.sigma, self.dt, np.array(action),dP)
        next_time = np.ones_like(next_wealth)*state[0]+self.dt

        done = next_time >= self.T
        reward = (done)*self.U_2(self.V_t)

        # Additional info (not used for now)
        info = {}

        return np.array([next_time, next_wealth], dtype=np.float32), reward, done, dP


    def step(self, action):
        """
        Execute one time step within the environment.
        
            Parameters:
                action : investment in risky asset
            
            Returns:
                The tuple (state(next state,next time),reward,isTerminalState,shock return involved)            
       """
        # Update Wealth (see wealth dynamics, Inv. Strategies script (by Prof. Zagst) Theorem 2.18):
        dP = self.generate_random_returns(1).ravel()
        
        # Wealth process update via simulation of the exponent
        self.V_t = self.V_t *self.wealth_update(self.r, self.mu, self.sigma, self.dt, np.array(action), dP)
        self.t += self.dt

        done = self.t >= self.T
        reward = (done)*self.U_2(self.V_t)

        # Additional info (not used for now)
        info = dP

        return self._get_obs(), reward, done, dP

    def VU(self,v,a,dP):
        """
        Updates a complete mini batch of (state,action). dP is a vector of log returns, sampled from replay bufffer or could be generated from a distribution (See the Estimate version ). So in effect for each of the allocations in action, it takes all the returns specified in dP and computes a cross product of them. Very similar to the grid update but uses internal variables for the missing input.

        Parameters:
            v - Intial wealth
            a - action 
            dP - log shock retruns of the risky asset
        
        Returns:
            The updated wealth of each action across all the shock returns. If the size of the actions is n and size of shock returns is m, then the resulting wealth update will be of the dimension m * n

        """
        return v*tf.exp(((1-a)*self.r*self.dt) +a*dP +  0.5*a*(1-a)*self.dt*(self.sigma**2))

    def rw(self,t_1,Vu):
        done = tf.cast(t_1 >= self.T,tf.float32)
        return  (done)*self.U_2(Vu)


    def _get_obs(self):
        """Internal function - Not needed """
        return np.array([self.t, self.V_t], dtype=np.float32)

    def reset(self):
        """Reset the state of the environment to an initial state. Used in the beginning of every episode"""

        self.t = np.zeros_like(self.V_0)
        if isinstance(self.V_0, str):
            self.V_t = np.array(eval(self.V_0)).reshape(-1)
        else:
            self.V_t = self.V_0

        return self._get_obs()

    def render(self, mode='human', close=False):
        """
            Not used currently
        """
        # Render the environment to the screen
        raise NotImplementedError("Use the discrete_bs_render environment for rendering")