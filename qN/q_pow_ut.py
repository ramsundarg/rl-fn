
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
class Q:
    def __init__(self,cfg):
        
        self.m=cfg['ddpg']['q']['variables']['m']
        self.s=cfg['ddpg']['q']['variables']['s']
        self.variables ={}
        self.variables['actual'] = [tf.Variable(self.m,dtype="float32"),tf.Variable(self.s,dtype="float32")]
        self.variables['target'] =[self.m,self.s]
        self.cfg = cfg
        self.r = self.cfg['env']['r']
        self.b = self.cfg['env']['b']
        self.T = self.cfg['env']['T']
        self.dt = self.cfg['env']['dt']
        
        
    def q_mu(self,arr,network='actual'):
        t = tf.reshape(np.array(arr[0])[:,0],[-1,1])
        V = tf.reshape(np.array(arr[0])[:,1],[-1,1])
        a = tf.cast(tf.reshape(arr[1],[-1,1]),tf.float32)
        b = self.b
        T = self.T
        r= self.r
        dt = self.dt
        mu = self.variables[network][0]
        sigma = self.variables[network][1]
        r_ex = mu - r
        sigma_2 = sigma ** 2
        t_i_plus_1 = T - t - dt
        val = tf.cast(tf.exp(b * (r + ((r_ex ** 2) / ((1 - b) * sigma_2)) * (1 -0.5/(1 - b))) * t_i_plus_1),tf.float32) \
                *    tf.cast(tf.exp(0.5 * ((b / (1 - b)) ** 2) * ((r_ex ** 2/ sigma_2) ) * t_i_plus_1),tf.float32)  \
                * tf.cast(tf.exp((b * r +  b*a * r_ex + 0.5 * (b*(b-1)) * (a ** 2) * sigma_2) * dt),tf.float32) 
        return tf.where(t<T, tf.pow(V, b) / b * val,0) 

    def get_all_variables(self):
        return self.variables['target'],self.variables['actual']
    
    def get_trainable_variables(self):
        return self.variables['actual']