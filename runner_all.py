# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:41:24 2022

@author: A00006846
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:51:27 2022

@author: A00006846
"""

# from plotting.action_value_function_power_utility import plot_optimal_q_value_surface_pow_ut




import pandas as pd
from collections.abc import MutableMapping
import itertools
import mlflow
import yaml
import json
from ddpg_generic import DDPGAgent
import numpy as np
import importlib
import copy
from mlflow import log_metric, log_param, log_artifact, tensorflow
def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict


def trainer(env, agent, max_episodes, max_steps, batch_size, action_noise, adamp, bsize_increase="0"):
    episode_rewards = []
    c = 0
    start_update = False
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        bsize = 0
        if isinstance(batch_size, str):
            bsize = eval(batch_size)
        else:
            bsize = batch_size
        i = 1
        for step in range(max_steps):
            c = c+1
            action = agent.get_action(state, action_noise)
            next_state, reward, done, _ = env.step(action)
            d_store = False if step == max_steps - 1 else done
            agent.replay_buffer.push(
                state, action, reward, next_state, d_store)
            episode_reward += reward

            if agent.replay_buffer.size > bsize:
                agent.update(bsize)
                bsize = bsize+int(eval(bsize_increase))
                i = i+1
                start_update = True

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                if start_update:
                    agent.log_metrics(c)
                if episode % 100 == 0:
                    print("Episode " + str(episode) +
                          ": " + str(episode_reward))
                break

            state = next_state
            action_noise = action_noise * adamp

    if start_update:
        agent.log_metrics(c)
    return episode, episode_rewards




def run_experiment(cfg):

    with mlflow.start_run(run_name=cfg.get('name', "Default Run"),experiment_id=experiment.experiment_id):
        #tensorflow.autolog()
        d = flatten_dict(cfg)
        for k, v in d.items():
            if isinstance(v, str) == False and isinstance(v, list) == False:
                log_metric(k, v)
            else:
                log_param(k,v)

        mlflow.log_dict(cfg, "cur_cfg.txt")

        env_lib = importlib.import_module('{}'.format(cfg['env']['name']))
        env = getattr(env_lib, cfg['env']['name'])(cfg['env'])

        env_cfg = cfg['env']
        mu, sigma, r, T, dt, V_0 = env_cfg['mu'], env_cfg['sigma'], env_cfg[
            'r'], env_cfg['T'], env_cfg['dt'], env_cfg['V_0']
        setting = cfg['general_settings']
        max_episodes = setting['max_episodes']
        max_steps = setting['max_steps']
        batch_size = setting['batch_size']

        ddpg_settings = cfg['ddpg']
        gamma = ddpg_settings['gamma']
        tau = ddpg_settings['tau']
        buffer_maxlen = ddpg_settings['buffer_len']
        critic_lr = ddpg_settings['q']['lr']
        actor_lr = ddpg_settings['a']['lr']
        adamp = ddpg_settings.get("action_noise_damp", 1.0)

        qN_lib = importlib.import_module(
            'qN.{}'.format(cfg['ddpg']['q']['name']))
        qN = getattr(qN_lib, 'Q')(cfg)

        aN_lib = importlib.import_module(
            'aN.{}'.format(cfg['ddpg']['a']['name']))
        aN = getattr(aN_lib, 'A')(cfg)

        agent = DDPGAgent(env, ddpg_settings, qN, aN)
        episodes, episode_rewards = trainer(
            env, agent, max_episodes, max_steps, batch_size, 0.01, adamp, setting.get("batch_size_increase", "i"))

        # wealths = np.linspace(0, 1, 20, dtype='float32')
        # risky_asset_allocations = np.linspace(-1, 1, 41, dtype='float32')
        # times = np.linspace(0, T, 11, dtype='float32')

        log_param("q_variables", [x for x in list(
            itertools.chain(*qN.get_all_variables()))])
        log_param("a_variables", [x for x in list(
            itertools.chain(*aN.get_all_variables()))])

        def plot(v, title, legends=()):
            from matplotlib import pyplot as plt
            lineObjects = plt.plot(v)
            plt.title(title)
            if(len(legends) > 0):
                plt.legend(iter(lineObjects), legends)
            plt.show()

        # plot(agent.storage,title = "variables",legends =("mu","sigma"))
        # plot(agent.q_losses,title = "losses")
        # plot(episode_rewards,title = "rewards")
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
mlflow.set_experiment(cfg["name"])
experiment = mlflow.get_experiment_by_name(cfg["name"])
hypertune(0, 0, cfg_copy.get('tune', []))



