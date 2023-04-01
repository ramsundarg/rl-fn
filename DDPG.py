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

import CommonBuffer
import CommonDDPG
#from ddpg_generic import DDPGAgent

class DDPG(CommonDDPG.DDPG):
    def __init__(self,cfg):
        # Number of "experiences" to store at max
        self.cfg = cfg
        attr_dict = { 'state'  : 2 , 'action' : 1 , 'reward' : 1 , 'next_state' : 2} # Value of each key correponds to dimensions (shape) of the attribute 
        super().__init__(cfg)
        self.buffer = CommonBuffer.CommonBuffer(cfg, attr_dict)


    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        obs_dict = { 'state' : obs_tuple[0],'action' : obs_tuple[1], 'reward' : obs_tuple[2],'next_state' : obs_tuple[3]}
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
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = aN.mu(next_state_batch, 'target')
            y = tf.cast(reward_batch,tf.float32) + gamma * qN.q_mu(
                [next_state_batch, target_actions], 'target'
            )
            critic_value = qN.q_mu([state_batch, action_batch], 'actual')
            self.critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))


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
    def learn(self,episode):
        # Get sampling range
        attr_dict = self.buffer.get_batch(['state','action','reward','next_state'])
        super().learn(attr_dict)


