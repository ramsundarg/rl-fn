# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import importlib
import BSAvgState
#from ddpg_generic import DDPGAgent

class CommonBuffer:

    def __init__(self,cfg,attr_dict):
        # Number of "experiences" to store at max
        self.cfg = cfg
        self.buffer_capacity = cfg['ddpg']['buffer_len']
        # Num of tuples to train on.
        self.batch_size = cfg['general_settings']['batch_size']
        self.batch_increase = cfg['general_settings'].get("batch_size_increase", "none")
        self.episode_count = 1

        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        
        self.buffer = dict()
        for key in attr_dict.keys():
            self.buffer[key] = np.zeros((self.buffer_capacity,attr_dict[key]))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_dict):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        for key in obs_dict:
            self.buffer[key][index] = obs_dict[key]
        self.buffer_counter += 1
        
    def get_batch(self,attr_list,batch_size=None,increase = True):
        if batch_size == None:
            batch_size = self.batch_size
        if self.batch_increase == "linear" and increase and self.buffer_counter > self.batch_size:
            batch_size = batch_size + (self.buffer_counter-self.batch_size)
        if self.batch_increase == "quadratic" and increase and self.buffer_counter > self.batch_size:
                batch_size = batch_size + np.sqrt(self.buffer_counter-self.batch_size)
        if batch_size >  int(0.7*self.buffer_capacity):
            batch_size = int(0.7*self.buffer_capacity)
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, batch_size)
        attr_dict =dict()
        for key in attr_list:
            batch = tf.convert_to_tensor(self.buffer[key][batch_indices])                
            attr_dict[key] = batch
        return attr_dict