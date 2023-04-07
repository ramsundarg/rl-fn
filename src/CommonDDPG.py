import tensorflow as tf
import numpy as np
import importlib
import BSAvgState
import CommonBuffer
# -*- coding: utf-8 -*-

class DDPG:
    """
        The common DDPG class that has to be imported if you need to customize a new DDPG method.  Please read the usage page in the docs on how to add a new DDPG class. These methods are the common methods that help to quickly prototype the new class.
    """
    def __init__(self,cfg):
        """
        Initializes a DDPG object. Please look at a sample cfg file to see how you can use the file to initialize settings here.
        Note that the modules for Q and A should be actual modules defined in qN and aN folders respectively.
        
        Parameters:
        cfg - The config file that is used as input to run experiments.
        """
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
        self.critic_loss = 1
        self.factor =1 
    
    def update_target_weights(self,N,tau):
        """
            Updates the target network. Could be Q or A. The rate tau can also be specified. Note that if the network has the attribute update_weight set, then this function is skipped and instead the networks function is called. It is useful if you need any customizations to the target network.

            Parameters:
                N - The network q or a
                tau - The learning rate for the target network
             
        """
        if hasattr(N,'update_weight'):
            return N.update_weight()
        t,a = N.get_all_variables()
        for i in range(len(t)):
            t[i] =tau*a[i] + (1-tau)*t[i]
            
    def learn(self,attr_dict):
        """
            The learning step of DDPG. It calls the derived class of DDPG's update method. The update method is not defined in this case but it is left to individual implementation of DDPG to redefine it to provide flexibility. Then it updates the target networks as specified in the update_target_weights function.

            Parameters:
                attr_dict: The parameters required for the learning. Could be the usual state,action,reward,done tuple or additional parameters that have to be trained.

        """
        self.update(attr_dict)
        self.update_target_weights(self.aN, self.tau*self.factor)
        self.update_target_weights(self.qN,  self.tau*self.factor)

    def update_tau(self,factor):
        """Simple function that updates factor for tau."""
        self.factor = float(factor)


