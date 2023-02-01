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
import tensorflow-probability as tfp

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
        tfd = tfp.distributions
        self.dist = tf.Normal(0,1)

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
            dW_t = tf.random.normal(shape=(state_batch.shape[0]+1,self.m))

            diff = dW_t[1:,:]-dW_t[:-1,:]
            dW_t = dW_t[:-1,:]
            
            action= tf.repeat(action_batch,dW_t.shape[1],axis=1)
            wealth = tf.repeat(state_batch[:,0],dW_t.shape[1],axis=1)
            time = tf.repeat(state_batch[:,1],dW_t.shape[1],axis=1)
            wealth_paths = env.VU(state_batch,action_batch,dW_t) #shape : (s,z)
            t_1 = time+env.dt
            rewards= env.r(t_1,wealth_paths)
            pd= self.dist.prob(dW_t)
            ns =  tf.stack([tf.cast(t_1,tf.float32),wealth_paths],axis=2)
            target_actions = aN.mu(ns, 'target') #The dimensions is a big array of [state_buffer_size * shock_batch_size,1]
            q_next = (rewards + tf.reshape(qN.q_mu([ns,target_actions], 'target'),wealth_paths.shape))*pd*diff
            q_next = tf.math.reduce_sum(q_next,axis=1,keepdims=True) #Averaged per shock_batch
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

