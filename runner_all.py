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
import yaml
import json

from ddpg_generic import DDPGAgent
import numpy as np
import importlib

#from plotting.action_value_function_power_utility import plot_optimal_q_value_surface_pow_ut






def trainer(env, agent, max_episodes, max_steps, batch_size, action_noise):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, action_noise)
            next_state, reward, done, _ = env.step(action)
            d_store = False if step == max_steps - 1 else done
            agent.replay_buffer.push(state, action, reward, next_state, d_store)
            episode_reward += reward

            if agent.replay_buffer.size > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                if episode % 100 == 0:
                    print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode,episode_rewards

cfg = {}
with open("cfg/powut1.json","r") as jfile:
    cfg = json.load(jfile)


env_lib = importlib.import_module('{}'.format(cfg['env']['name'])) 
env  = getattr(env_lib, cfg['env']['name'])(cfg['env'])

env_cfg = cfg['env']
mu, sigma, r, T, dt, V_0 = env_cfg['mu'],env_cfg['sigma'],env_cfg['r'],env_cfg['T'],env_cfg['dt'],env_cfg['V_0']
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

qN_lib = importlib.import_module('qN.{}'.format(cfg['ddpg']['q']['name'])) 
qN  = getattr(qN_lib, 'Q')(cfg)

aN_lib = importlib.import_module('aN.{}'.format(cfg['ddpg']['a']['name'])) 
aN  = getattr(aN_lib, 'A')(cfg)

agent = DDPGAgent(env, ddpg_settings,qN,aN)
episodes,episode_rewards = trainer(env, agent, max_episodes, max_steps, batch_size, action_noise=0.01)

wealths = np.linspace(0, 1, 20, dtype='float32')
risky_asset_allocations = np.linspace(-1, 1, 41, dtype='float32')
times = np.linspace(0, T, 11, dtype='float32')




def plot(v,title,legends=()):
    from matplotlib import pyplot as plt
    lineObjects= plt.plot(v)
    plt.title(title)
    if(len(legends)>0):
        plt.legend(iter(lineObjects), legends)
    plt.show()
    
 
plot(agent.storage,title = "variables",legends =("mu","sigma"))
plot(agent.q_losses,title = "losses")
plot(episode_rewards,title = "rewards")