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

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        obs_dict = { 'state' : obs_tuple[0],'action' : obs_tuple[1], 'reward' : obs_tuple[2],'next_state' : obs_tuple[3],'shock' : obs_tuple[4]}
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
        shock_batch = attr_dict['shock']
        aN= self.aN
        qN = self.qN
        gamma= self.gamma
        env = self.env
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            
            critic_value = qN.q_mu([state_batch, action_batch], 'actual') #Dimensions state_batch_size
            dP = shock_batch #Randomly polled from the shock buffer , has no corresponding indices with state buffer. Is this correct ?
            next_returns = env.wealth_grid_update(env.r, env.mu, env.sigma, env.dt, action_batch, dP) #This is a big update with final dimensions (state_buffer_size X shock_batch_size)
            wealth_paths = tf.cast(tf.multiply(next_returns, (state_batch[:,1])[:,tf.newaxis]),dtype=tf.float32) #Dimensions would be (state_buffer_size X shock_batch_size)
            t_1 = tf.repeat((state_batch[:,0]+env.dt)[:,tf.newaxis],dP.shape[0],axis=1) #Computing next time because reward batch is now recomputed.
            #t_0 = tf.repeat((state_batch[:,0])[:,tf.newaxis],dP.shape[0],axis=1) #Computing next time because reward batch is now recomputed.
            done = tf.cast(t_1 >= env.T,tf.float32)
            ns =  tf.stack([tf.cast(t_1,tf.float32),wealth_paths],axis=2)
            target_actions = aN.mu(ns, 'target') #The dimensions is a big array of [state_buffer_size * shock_batch_size,1]
            q_next = (done)*env.U_2(wealth_paths) + tf.reshape(qN.q_mu([ns,target_actions], 'target'),wealth_paths.shape)
            q_next = tf.math.reduce_mean(q_next,axis=1,keepdims=True) #Averaged per shock_batch
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
        temp = self.buffer.get_batch(['shock'],self.m,False)
        attr_dict['shock']= temp['shock']
        super().learn(attr_dict)

