# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
class A:
    def get_model(self,cfg):
        input_x = Input(shape=2)
        self.custom_weight = True
        hidden_layers = cfg['ddpg']['a']['variables']
        self.tau = cfg['ddpg']['tau']
        initializer = tf.keras.initializers.GlorotUniform()
        x = input_x
        for i in hidden_layers:
            x = Dense(i,activation='relu', kernel_initializer=initializer)(x)
        x = Dense(1,activation='relu', kernel_initializer=initializer)(x)
        x = tf.math.multiply(x,cfg['ddpg'].get('action_mult',1))
        self.update_actor = True
        return  tf.keras.Model(input_x,x)
    
    def __init__(self,cfg):
        self.network = {}
        self.network['actual'] = self.get_model(cfg)
        self.network['target'] = self.get_model(cfg)
        
        
    def mu(self,X,network_str= 'actual'):
        model = self.network[network_str]
        a = model(X)
        return a
        
    def custom_update(self,ddpg_agent):
        for i in range(len(self.variables['actual'])):
            self.variables['actual'][i]=ddpg_agent.qN.variables['actual'][i]
            self.variables['target'][i]=ddpg_agent.qN.variables['target'][i]

    def get_all_variables(self):
        return self.network['target'].get_weights(),self.network['actual'].get_weights()
    
    def get_trainable_variables(self):
        return self.network['actual'].trainable_variables

    def update_weight(self):
        temp1 = np.array(self.network['target'].get_weights())
        temp2 = np.array(self.network['actual'].get_weights())
        temp3 = self.tau*temp2 + (1-self.tau)*temp1
        self.network['target'].set_weights(temp3)