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
#from ddpg_generic import DDPGAgent

class Buffer:
    def __init__(self,cfg):
        # Number of "experiences" to store at max
        self.cfg = cfg
        self.buffer_capacity = cfg['ddpg']['buffer_len']
        # Num of tuples to train on.
        self.batch_size = cfg['general_settings']['batch_size']
        self.m = cfg['buffer']['m']
        
        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, 2))
        self.action_buffer = np.zeros((self.buffer_capacity, 1))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, 2))
        
        self.rs = np.zeros((self.buffer_capacity, self.m))
        self.vs = np.zeros((self.buffer_capacity, self.m))
        qN_lib = importlib.import_module(
            'qN.{}'.format(cfg['ddpg']['q']['name']))
        qN = getattr(qN_lib, 'Q')(cfg)

        aN_lib = importlib.import_module(
            'aN.{}'.format(cfg['ddpg']['a']['name']))
        aN = getattr(aN_lib, 'A')(cfg)
        self.aN = aN
        self.qN = qN
        self.gamma = cfg['ddpg']['gamma']
        self.tau = cfg['ddpg']['tau']
        self.critic_optimizer = tf.keras.optimizers.Adam(cfg['ddpg']['q']['lr'])
        self.actor_optimizer = tf.keras.optimizers.Adam(cfg['ddpg']['a']['lr'])
        self.actor_loss = 1
        self.env  = BSAvgState.BSAvgState(cfg['env'])


    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        next_state,reward,done,_ = self.env.peek_steps(obs_tuple[0], obs_tuple[1], self.m)
        self.rs[index] = reward.concat(obs_tuple[2])
        self.vs[index] = next_state[1].concat(obs_tuple[3][1])

        self.buffer_counter += 1

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

    def update_target_weights(self,N,tau):
        if hasattr(N,'update_weight'):
            return N.update_weight()
        t,a = N.get_all_variables()
        for i in range(len(t)):
            t[i] =tau*a[i] + (1-tau)*t[i]
    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        vs_batch = tf.convert_to_tensor(self.vs[batch_indices])
        rs_batch = tf.convert_to_tensor(self.rs[batch_indices])
        self.update(state_batch, action_batch, reward_batch, next_state_batch,vs_batch,rs_batch)
        self.update_target_weights(self.aN, self.tau)
        self.update_target_weights(self.qN,  self.tau)
