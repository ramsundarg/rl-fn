# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:45:54 2022

@author: A00006846
"""


"""
The `Buffer` class implements Experience Replay.
---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---
**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.
**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.
Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""

import tensorflow as tf

import numpy as np
import importlib
import BSAvgState
import CommonBuffer
import CommonDDPG
from statistics import NormalDist
#from ddpg_generic import DDPGAgent

class DDPG(CommonDDPG.DDPG):
    def __init__(self,cfg):
        # Number of "experiences" to store at max
        self.cfg = cfg
        self.m = cfg['buffer']['m']
        attr_dict = { 'state'  : 2 , 'action' : 1 , 'reward' : 1 , 'next_state' : 2, 'shock' : 1} # Value of each key correponds to dimensions (shape) of the attribute 
        super().__init__(cfg)
        self.buffer = CommonBuffer.CommonBuffer(cfg, attr_dict)
        self.env  = BSAvgState.BSAvgState(cfg['env'])
        self.sdt = tf.sqrt(self.env.dt)
        
        n=self.m
        self.n = n
        self.z = np.array([NormalDist(0,self.sdt).inv_cdf((n+1+i)/(2*(n+1))) for i in range(-n,n+1)])
        self.dP_v =  (self.env.mu - 0.5*self.env.sigma**2)*self.env.dt + self.env.sigma* self.z
        self.pd_v=  (np.vectorize(NormalDist(0,self.sdt).pdf))(self.z)
        self.diff_orig = [0]+[self.z[i+1]-self.z[i] for i in range(len(self.z)-1)]
        
        self.diff = tf.tile([self.diff_orig],[cfg['general_settings']['batch_size'],1])
        self.dW_t = tf.cast(tf.tile([self.z],[cfg['general_settings']['batch_size'],1]),tf.float32)
        self.dP =  tf.cast(tf.tile([self.dP_v],[cfg['general_settings']['batch_size'],1]),tf.float32)
        self.pd=  tf.cast(tf.tile([self.pd_v],[cfg['general_settings']['batch_size'],1]),tf.float32)

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        obs_dict = { 'state' : obs_tuple[0],'action' : obs_tuple[1], 'reward' : obs_tuple[2],'next_state':  obs_tuple[3] }
        self.buffer.record(obs_dict)

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    #@tf.function
    def update(
        self,attr_dict
        
    ):
        state_batch = attr_dict['state'] 
        action_batch = attr_dict['action']
        reward_batch = attr_dict['reward']
        next_state_batch = attr_dict['next_state']
        aN= self.aN
        qN = self.qN
        gamma= self.gamma
        env = self.env
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            
            critic_value = qN.q_mu([state_batch, action_batch], 'actual') #Dimensions state_batch_size
            
            if self.dW_t.shape[0]!= state_batch.shape[0]: # Can happen if batch sizes change over episodes
                self.diff = tf.tile([self.diff_orig],[state_batch.shape[0],1])
                self.dW_t = tf.cast(tf.tile([self.z],[state_batch.shape[0],1]),tf.float32)
                self.dP =  tf.cast(tf.tile([self.dP_v],[state_batch.shape[0],1]),tf.float32)
                self.pd=  tf.cast(tf.tile([self.pd_v],[state_batch.shape[0],1]),tf.float32)

                
            dW_t = self.dW_t
            diff = self.diff
            dP=self.dP
            pd=self.pd
            
            
                
            
            
            action= tf.cast(tf.repeat(action_batch,dW_t.shape[1],axis=1),tf.float32)
            wealth = tf.cast(tf.repeat(tf.reshape(state_batch[:,1],[-1,1]),dW_t.shape[1],axis=1),tf.float32)
            time = tf.cast(tf.repeat(tf.reshape(state_batch[:,0],[-1,1]),dW_t.shape[1],axis=1),tf.float32)
            wealth_paths = env.VU(wealth,action,dP) #shape : (s,z)
            t_1 = time+env.dt
            rewards= env.rw(t_1,wealth_paths)
            
            ns =  tf.stack([tf.cast(t_1,tf.float32),wealth_paths],axis=2)
            target_actions = aN.mu(ns, 'target') #The dimensions is a big array of [state_buffer_size * shock_batch_size,1]
            q_next = (rewards + tf.reshape(qN.q_mu([ns,target_actions], 'target'),wealth_paths.shape))*pd*diff
            q_next = tf.math.reduce_sum(q_next,axis=1,keepdims=True) #Averaged per shock_batch
            q_next = q_next #/ pd_sum
            self.critic_loss = tf.math.reduce_mean(tf.math.square(q_next - critic_value))
           

        critic_grad = tape.gradient(self.critic_loss, qN.get_trainable_variables())
        self.critic_optimizer.apply_gradients(
            zip(critic_grad,  qN.get_trainable_variables())
        )
        if aN.update_actor:
            with tf.GradientTape() as tape:
                actions = aN.mu(state_batch, 'actual')
                critic_value = qN.q_mu([state_batch, actions], 'actual')
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                self.actor_loss = -tf.math.reduce_mean(critic_value)
    
    
                #mlflow.log_metric("A loss",actor_loss.numpy())
    
            actor_grad = tape.gradient(self.actor_loss, aN.get_trainable_variables())
            self.actor_optimizer.apply_gradients(
                zip(actor_grad,  aN.get_trainable_variables())
            )
        else:
            aN.custom_update(self)



    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        attr_dict = self.buffer.get_batch(['state','action','reward','next_state'])
        super().learn(attr_dict)

