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
        attr_dict = { 'state'  : 2 , 'action' : 1 , 'reward' : 1 , 'next_state' : 2, 'vs' : self.m , 'rs' : self.m } # Value of each key correponds to dimensions (shape) of the attribute 
        super().__init__(cfg)
        self.buffer = CommonBuffer.CommonBuffer(cfg, attr_dict)
        self.env  = BSAvgState.BSAvgState(cfg['env'])


    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        obs_dict = { 'state' : obs_tuple[0],'action' : obs_tuple[1], 'reward' : obs_tuple[2],'next_state' : obs_tuple[3]}
        next_state,reward,done,_ = self.env.peek_steps(obs_tuple[0], obs_tuple[1], self.m)
        obs_dict['vs'] = next_state
        obs_dict['rs'] = reward
        self.buffer.record(obs_dict)



    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    #@tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        vs_batch,
        rs_batch
        
    ):
        aN= self.aN
        qN = self.qN
        gamma= self.gamma
        env =self.env
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            critic_value = qN.q_mu([state_batch, action_batch], 'actual')
            wealth_paths =  tf.cast(vs_batch,tf.float32)#Dimensions would be (batch size X )
            t_1 = tf.repeat((state_batch[:,0]+env.dt)[:,tf.newaxis],vs_batch.shape[1],axis=1) #Computing next time because reward batch is now recomputed.
            #t_0 = tf.repeat((state_batch[:,0])[:,tf.newaxis],dP.shape[0],axis=1) #Computing next time because reward batch is now recomputed.
            ns =  tf.stack([tf.cast(t_1,tf.float32),wealth_paths],axis=2)
            target_actions = aN.mu(ns, 'target') #The dimensions is a big array of [state_buffer_size * shock_batch_size,1]
            q_next = tf.cast(rs_batch,tf.float32) + tf.reshape(qN.q_mu([ns,target_actions], 'target'),wealth_paths.shape)
            q_next = tf.math.reduce_mean(q_next,axis=1,keepdims=True) #Averaged per shock_batch

            self.critic_loss = tf.math.reduce_mean(tf.math.square(q_next - critic_value))


        critic_grad = tape.gradient(self.critic_loss, qN.get_trainable_variables())
        self.critic_optimizer.apply_gradients(
            zip(critic_grad,  qN.get_trainable_variables())
        )

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


    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        attr_dict = self.buffer.get_batch(['state','action','reward','next_state','vs','rs'])
        super().learn(attr_dict)
