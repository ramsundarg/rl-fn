# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:38:02 2022

@author: A00006846
"""

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

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




"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically
an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.
"""


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
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
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)







"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(sampled_actions, noise_object,factor=1,scale=1):
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + scale*factor*noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, -1, 1)

    return [np.squeeze(legal_action)]




std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

def trainer(cfg):
    
    total_episodes = cfg['general_settings']['max_episodes']
    buffer_lib = importlib.import_module('{}'.format(cfg['buffer']['name']))
    buffer = getattr(buffer_lib, "DDPG")(cfg)
    env_lib = importlib.import_module('{}'.format(cfg['env']['name']))
    env = getattr(env_lib, cfg['env']['name'])(cfg['env'])
    tau_decay = (cfg['ddpg'].get('tau_decay',"none")=="linear")
    a_vals = np.zeros(1000)
    for ep in range(total_episodes):
    
        prev_state = env.reset()
        episodic_reward = 0
    
        while True:
    
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            sampled_actions = tf.squeeze(buffer.aN.mu(tf_prev_state,'actual'))

            factor = cfg['ddpg'].get('noise_decay',1)
            if factor == "linear":
                factor = (total_episodes-ep)*1.0/total_episodes
                
            action = policy(sampled_actions, ou_noise,factor,cfg['ddpg'].get('noise_scale',1))
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)
    
            buffer.record((prev_state, action, reward, state,info))
            episodic_reward += reward
    
            buffer.learn()
            if tau_decay:
                buffer.update_tau((total_episodes-ep)*1.0/total_episodes)

            # End this episode when `done` is True
            if done:
                break
    
            prev_state = state
            a_vals[ep%1000]=buffer.aN.mu(np.expand_dims([0,cfg['env']['V_0']], axis=0),'target').numpy() 
        if ep%100==0:
            mlflow.log_metric("Factor",factor,ep)
            mlflow.log_metric("A_Value",buffer.aN.mu(np.expand_dims([0,cfg['env']['V_0']], axis=0),'target').numpy(),ep)
            mlflow.log_metric("A_Value_Smooth",np.mean(a_vals),ep)
            mlflow.log_metric("A_Value_Variance",np.var(a_vals),ep)
            mlflow.log_metric("A loss",buffer.actor_loss.numpy(),ep)
            mlflow.log_metric("Q loss",buffer.critic_loss.numpy(),ep)
            
    log_param("q_variables", [x.numpy() for x in list(
        itertools.chain(*buffer.qN.get_all_variables()))])
    log_param("a_variables", [x.numpy() for x in list(
        itertools.chain(*buffer.aN.get_all_variables()))])
    mlflow.log_param("Accuracy",(np.mean(a_vals)/cfg['A_Value_Ex']))
    





def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict
# Save the weights
def run_experiment(cfg):
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
