# -*- coding: utf-8 -*-
"""
The main runner of the whole algorithm

"""

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


import pandas as pd
from collections.abc import MutableMapping
import itertools
import mlflow
import yaml
import json
import numpy as np
import importlib
import copy
import os
#from NewBuffer import Buffer
from mlflow import log_metric, log_param, log_artifact, tensorflow

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




class OUActionNoise:
    """
        To implement better exploration by the Actor network, we use noisy perturbations,
        specifically an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.

        It samples noise from a correlated normal distribution.
    """

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        """
            Initializes the noise process. 

            Parameters:

                mean - The mean of the noise

                std_deviation  - standard deviation of the noise

                theta - The reverting factor towards the mean

                dt - time step

                x_initial - Intial value of the noise level.

        """
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        """
             Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process. This function is called every step to generate some level of noise to be added to the optimal action to improve the exploration part of the algorithm.

        
        """
        
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        """
            This method resets the noise level back after every episode.

        """
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)







"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(sampled_actions, noise_object,factor=1,scale=1):
    """
        The policy method adds the noise to the polled actions. The OU process can be one such noise process that can be added to the action value. The policy also dictates how to scale and factor the noise over the course of an experiment.

        Parameters:

            sampled_actions: The polled actions from the replay buffer.

            noise_object: Could be any model. We have implemented an OU process. It should implement the _call__ method.

            factor - The factor of the noise level to be added depending on the stage of the experiment. for example we can decay noise over episodes.

            scale - This controls the multiplier of the noise level obtained by the noise process. 


        Returns:

            The noise to be mixed to the sampled actions.

    """
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + scale*factor*noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, -1, 1)
    legal_action = sampled_actions

    return [np.squeeze(legal_action)]




std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

def trainer(cfg):
    """
        The runner of the whole algorithm. This method iterates over the episodes and for every step calls the environment to transition to further states and then stores the new transitions into a replay buffer. Finally the DDPG is updated at every iteration and the losses are recorded after every step.

        Parameters:

            cfg - The configuration of the experiment.
    """
    total_episodes = cfg['general_settings']['max_episodes']
    buffer_lib = importlib.import_module('{}'.format(cfg['buffer']['name']))
    ddpg = getattr(buffer_lib, "DDPG")(cfg)
    env_lib = importlib.import_module('{}'.format(cfg['env']['name']))
    env = getattr(env_lib, cfg['env']['name'])(cfg['env'])
    tau_decay = (cfg['ddpg'].get('tau_decay',"none")=="linear")
    a_vals = np.zeros(1000)
    for ep in range(total_episodes):
    
        prev_state = env.reset()
        episodic_reward = 0
    
        while True:
    
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            sampled_actions = tf.squeeze(ddpg.aN.mu(tf_prev_state,'actual'))

            factor = cfg['ddpg'].get('noise_decay',1)
            if factor == "linear":
                factor = (total_episodes-ep)*1.0/total_episodes
            
            
                
            action = policy(sampled_actions, ou_noise,factor,cfg['ddpg'].get('noise_scale',1))
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)
    
            ddpg.record((prev_state, action, reward, state,info))
            episodic_reward += reward
    
            ddpg.learn(ep)
            if tau_decay:
                ddpg.update_tau((total_episodes-ep)*1.0/total_episodes)

            # End this episode when `done` is True
            if done:
                break
    
            prev_state = state
            a_vals[ep%1000]=ddpg.aN.mu(np.expand_dims([0,cfg['env']['V_0']], axis=0),'target').numpy() 
        if ep%100==0:
            mlflow.log_metric("Factor",factor,ep)
            mlflow.log_metric("A_Value",ddpg.aN.mu(np.expand_dims([0,cfg['env']['V_0']], axis=0),'target').numpy(),ep)
            mlflow.log_metric("A_Value_Smooth",np.mean(a_vals),ep)
            mlflow.log_metric("A_Value_Variance",np.var(a_vals),ep)
            mlflow.log_metric("A loss",ddpg.actor_loss.numpy(),ep)
            mlflow.log_metric("Q loss",ddpg.critic_loss.numpy(),ep)
            if cfg['buffer'].get('decay',"none") != "none":
                mlflow.log_metric("Cur_M_Size",ddpg.n,ep)
            
    log_param("q_variables", [x.numpy() for x in list(
        itertools.chain(*ddpg.qN.get_all_variables()))])
    log_param("a_variables", [x.numpy() for x in list(
        itertools.chain(*ddpg.aN.get_all_variables()))])
    mlflow.log_param("Accuracy",(np.mean(a_vals)/cfg['A_Value_Ex']))
    





def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    """
        This is an internal method
    """
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict
# Save the weights
def run_experiment(cfg):
    """
            The wrapper for the trainer function. This just wraps mlflow utilities for storing the artifacts for the experiment.

            Parameter:
            
                cfg - The configuration of the experiment
    """
    experiment_name = '{}Type:{} q:{} a:{}'.format( cfg.get('name',''),cfg['env']["U_2"],cfg['ddpg']['q']['name'],cfg['ddpg']['a']['name'])
    env_cfg = cfg['env']

    if(env_cfg['U_2']=='np.log'):
        m = env_cfg['mu']
        s = env_cfg['sigma']
        r = env_cfg['r']
        cfg['A_Value_Ex'] = (m-r)/(s**2)
    else:
        m = env_cfg['mu']
        s = env_cfg['sigma']
        r = env_cfg['r']
        b = env_cfg['b']
        if b > 1:
            return
        cfg['A_Value_Ex'] = (m-r)/(s**2*(1-b))




    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run(run_name=experiment_name,experiment_id=experiment.experiment_id):
        d = flatten_dict(cfg)
        for k, v in d.items():
            if isinstance(v, str) == False and isinstance(v, list) == False:
                log_metric(k, v)
            else:
                log_param(k,v)

        mlflow.log_dict(cfg, "cur_cfg.txt")
        trainer(cfg)
        mlflow.end_run()
