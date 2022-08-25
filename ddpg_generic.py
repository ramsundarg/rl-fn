# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:49:34 2022

@author: A00006846
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from sys import exit
from buffer import BasicBuffer_a, BasicBuffer_b
import random
import json
from mlflow import log_metric, log_param, log_artifacts
# np.random.seed(0)
# tf.random.set_seed(0)


class DDPGAgent:
    
    def __init__(self, env, ddpg_settings,qN,aN):
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]
        # self.action_max = 1
        
        # hyperparameters
        self.gamma = ddpg_settings['gamma']
        self.tau = ddpg_settings['tau']
        buffer_maxlen = ddpg_settings['buffer_len']
        critic_lr = ddpg_settings['q']['lr']
        actor_lr = ddpg_settings['a']['lr']

        # optimizers
        self.qN = qN
        self.aN = aN
        self.q_mu_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.mu_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.replay_buffer = BasicBuffer_a(size = buffer_maxlen, obs_dim = self.obs_dim, act_dim = self.action_dim)
        self.q_losses = []
        self.mu_losses = []
        self.storage = []
        

 
    def update_target_weights(self,network='q'):
        t,a=None,None
        if network == 'q':
            t,a = self.qN.get_all_variables()
        else:
            t,a = self.aN.get_all_variables()
        for i in range(len(t)):
            t[i] =self.tau*a[i] + (1-self.tau)*t[i]
       
    
    def get_action(self, s, noise_scale):
        
        a = self.aN.mu(np.expand_dims(s, axis=0),'actual')
        a += noise_scale * np.random.randn(self.action_dim)
        return np.clip(a, -100, 100)

    def update(self, batch_size):
        X,A,R,X2,D = self.replay_buffer.sample(batch_size)
        X = tf.convert_to_tensor(X)
        A =  tf.convert_to_tensor(A)
        R =  tf.convert_to_tensor(R)
        X2 =  tf.convert_to_tensor(X2)

        # Updating  Critic
        with tf.GradientTape() as tape:
          A2 =  self.aN.mu(X2,'target')
          q_target = R + tf.reshape(self.gamma  * self.qN.q_mu([X2,A2],'target'),[-1,1])
          qvals = self.qN.q_mu([X,A]) 
          q_loss = tf.reduce_mean((qvals - q_target)**2)
          grads_q = tape.gradient(q_loss,self.qN.get_trainable_variables())
        self.q_mu_optimizer.apply_gradients(zip(grads_q, self.qN.get_trainable_variables()))
        log_metric("Q_loss",q_loss.numpy())
        self.q_losses.append(q_loss.numpy())
        self.storage.append([x.numpy() for x in self.qN.get_trainable_variables()])
        

        if self.aN.update_actor:
            with tf.GradientTape() as tape2:
              A_mu =  self.aN.mu(X)
              Q_mu = self.qN.q_mu([X,A_mu],'actual')
              mu_loss =  -tf.reduce_mean(Q_mu)
              grads_mu = tape2.gradient(mu_loss,self.aN.get_trainable_variables())
            self.mu_losses.append(mu_loss)
            log_metric("A_loss",mu_loss.numpy())
            log_metric("A_Value",self.aN.mu(np.expand_dims([0,0.5], axis=0)).numpy(),'target')
            self.mu_optimizer.apply_gradients(zip(grads_mu, self.aN.get_trainable_variables()))
            self.update_target_weights('a')
        else:
            self.aN.custom_update(self)
        

        
        self.update_target_weights('q')
      

