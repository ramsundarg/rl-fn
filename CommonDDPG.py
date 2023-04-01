import tensorflow as tf
import numpy as np
import importlib
import BSAvgState
import CommonBuffer
# -*- coding: utf-8 -*-

class DDPG:
    def __init__(self,cfg):
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
        if hasattr(N,'update_weight'):
            return N.update_weight()
        t,a = N.get_all_variables()
        for i in range(len(t)):
            t[i] =tau*a[i] + (1-tau)*t[i]
            
    def learn(self,attr_dict):
        self.update(attr_dict)
        self.update_target_weights(self.aN, self.tau*self.factor)
        self.update_target_weights(self.qN,  self.tau*self.factor)

    def update_tau(self,factor):
        self.factor = float(factor)


