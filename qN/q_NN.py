# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
class Q:
    def get_model(self,cfg):
        
        input_x = Input(shape=2)
        input_a = Input(shape=1)
        x = input_x
        initializer = tf.keras.initializers.GlorotUniform()
        hidden_layers = cfg['ddpg']['q']['variables']
        self.tau = cfg['ddpg']['tau']
        self.custom_weight = True
        for i,j in enumerate(hidden_layers[:-1]):
            if i==1:
                x = concatenate([x,input_a],axis=-1)
            x = Dense(j,activation='tanh',kernel_initializer=initializer)(x)
        x = Dense(hidden_layers[-1])(x)
        return tf.keras.Model([input_x,input_a],x)
    
    def __init__(self,cfg):
        self.network = {}
        self.network['actual'] = self.get_model(cfg)
        self.network['target'] = self.get_model(cfg)
        
        
    def q_mu(self,X,network_str= 'actual'):
        model = self.network[network_str]
        a = model(X)
        return a
        
    def get_all_variables(self):
        return self.network['target'].get_weights(),self.network['actual'].get_weights()
    
    def get_trainable_variables(self):
        return self.network['actual'].trainable_variables

    def update_weight(self):
        temp1 = np.array(self.network['target'].get_weights())
        temp2 = np.array(self.network['actual'].get_weights())
        temp3 = self.tau*temp2 + (1-self.tau)*temp1
        self.network['target'].set_weights(temp3)