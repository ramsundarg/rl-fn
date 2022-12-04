# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:41:16 2022

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
from NewBuffer import Buffer
from mlflow import log_metric, log_param, log_artifact, tensorflow
import run_experiment

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


def policy(sampled_actions, noise_object):
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, -1, 1)

    return [np.squeeze(legal_action)]




std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

def trainer(cfg):
    
    total_episodes = cfg['general_settings']['max_episodes']
    buffer_lib = importlib.import_module('{}'.format(cfg['buffer']['name']))
    buffer = getattr(buffer_lib, "Buffer")(cfg)
    env_lib = importlib.import_module('{}'.format(cfg['env']['name']))
    env = getattr(env_lib, cfg['env']['name'])(cfg['env'])
    for ep in range(total_episodes):
    
        prev_state = env.reset()
        episodic_reward = 0
    
        while True:
    
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            sampled_actions = tf.squeeze(buffer.aN.mu(tf_prev_state,'actual'))

    
            action = policy(sampled_actions, ou_noise)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)
    
            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward
    
            buffer.learn()

            # End this episode when `done` is True
            if done:
                break
    
            prev_state = state
            
        if ep%100==0:
            mlflow.log_metric("A_Value",buffer.aN.mu(np.expand_dims([0,100], axis=0),'target').numpy(),ep)
            mlflow.log_metric("A loss",buffer.actor_loss.numpy(),ep)
            mlflow.log_metric("Q loss",buffer.critic_loss.numpy(),ep)
            
    log_param("q_variables", [x.numpy() for x in list(
        itertools.chain(*buffer.qN.get_all_variables()))])
    log_param("a_variables", [x.numpy() for x in list(
        itertools.chain(*buffer.aN.get_all_variables()))])





def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict
# Save the weights
def run_experiment(cfg):
    experiment_name = 'Type:{} q:{} a:{}'.format( cfg['env']["U_2"],cfg['ddpg']['q']['name'],cfg['ddpg']['a']['name'])
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


pidx = 0
pitems = []
config_count = 0
def hypertune(level, idx, hypertune_items):
    global pidx
    global pitems,config_count,cfg_copy,cfg
    if idx == len(hypertune_items):
        if level == 0:
            cfg_copy.pop('tune', None)
            run_experiment(cfg_copy)
            
            print(cfg_copy)
            config_count= config_count+1
            print(config_count)
            print("\n\n\n")
            
        else:
            hypertune(level-1, pidx+1, pitems)
        
        return

    
    if (list(hypertune_items)[idx]) == "group":
        for el in list(hypertune_items.values())[idx]:
            pidx = idx
            pitems = hypertune_items
            hypertune(level+1, 0, el)
            for d in el.keys():
                keys =d.split('.')
                c = cfg_copy
                co = cfg
                update_key = True
                for key in keys[0:-1]:
                    if co.get(key, None) == None:
                        del c[key] 
                        update_key = False
                        break
                    if c.get(key, None) == None:
                        c[key] = []
                    co =co[key]
                    c= c[key]
                if update_key:
                    c[keys[-1]]= co[keys[-1]]
                
        return
    keys = (list(hypertune_items)[idx]).split('.')

    d = cfg_copy
    for key in keys[0:-1]:
        if d.get(key, None) == None:
            d[key] = []
        d = d.get(key)
    tune_item = list(hypertune_items.values())[idx]
    if(tune_item.get("low", "") != ""):
        is_value = False
        # print(tune_item["low"],tune_item["high"],tune_item["step"])
        for val in np.arange(tune_item["low"], tune_item["high"], tune_item["step"]):
            d[keys[-1]] = val.item()
            hypertune(level, idx+1, hypertune_items)
    if(tune_item.get("set")):
        for val in tune_item["set"]:
            d[keys[-1]] = eval(val)
            hypertune(level, idx+1, hypertune_items)
    if(tune_item.get("list")):
        for val in tune_item["list"]:
            d[keys[-1]] = val
            hypertune(level, idx+1, hypertune_items)
        



cfg = {}

file_name = "cfg/temp4.json"
with open(file_name, "r") as jfile:
    cfg = json.load(jfile)

cfg_copy = copy.deepcopy(cfg)
hypertune(0, 0, cfg_copy.get('tune', []))



