# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:45:54 2022

@author: Ramsundar
"""



import tensorflow as tf

import CommonBuffer
import CommonDDPG
#from ddpg_generic import DDPGAgent

class DDPG(CommonDDPG.DDPG):
    """
        The DDPG Functions version of the DDPG. Please read the thesis document to understand how it works.
    """
    def __init__(self,cfg):
        """
            A simple initialize function.  Note that we need to call the base class's init function here after we define the parameters we need to keep track for DDPG
        """
        # Number of "experiences" to store at max
        self.cfg = cfg
        attr_dict = { 'state'  : 2 , 'action' : 1 , 'reward' : 1 , 'next_state' : 2} # Value of each key correponds to dimensions (shape) of the attribute 
        super().__init__(cfg)
        self.buffer = CommonBuffer.CommonBuffer(cfg, attr_dict)


    
    def record(self, obs_tuple):
        """
            Records an observation from the experiment runner (run_experiment.py). We need to call this method and not the replay buffer's method because we need to translate the tuple to a dictionary we are interested in keeping track and then add additional variables here that can be passed to the replay buffer.

            Parameters:
                obs_tuple : The actual observations from the episode. The usual state,action,reward,done,info tuple generated from the BSAvgState environment.

        """
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
        """
            This is not called by the actual experiement runner in run_experiment.py.  The run_experiment makes a call to 'learn' method of this class instead. The learn method then calls the CommonDDPG.learn method, which in turn calls this update method. This workflow can be improved. But it is left as it is to provide flexibility for these methods while also retaining all common aspects in the base class. Note that when you design a new DDPG class, you should inherit from CommonDDPG and also must implement this method for the workflow to work.
            
            Parameters:
                obs_tuple : The actual observations from the episode. The usual state,action,reward,done,info tuple generated from the BSAvgState environment.

        """
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
        """
        The learn method called by run_experiment module. This method must be implemented for a  new DDPG class.
        Parameters:
            episode:  The episode number of the experiment.
        """
        # Get sampling range
        attr_dict = self.buffer.get_batch(['state','action','reward','next_state'])
        super().learn(attr_dict)


