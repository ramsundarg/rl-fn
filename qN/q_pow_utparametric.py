
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
class Q:
    def __init__(self,cfg):
        
        self.v=cfg['ddpg']['q']['variables']
        
        self.variables ={}
        self.variables['actual'] = [tf.Variable(self.v[0],dtype="float32"),tf.Variable(self.v[1],dtype="float32"),tf.Variable(self.v[2],dtype="float32"),tf.Variable(self.v[3],dtype="float32")]
        self.variables['target'] =[self.v[0],self.v[1],self.v[2],self.v[3]]
        self.cfg = cfg
        self.b = cfg['b']
        self.T = self.cfg['env']['T']
        self.dt = self.cfg['env']['dt']
        
        
    def q_mu(self,arr,network='actual'):
        t = tf.reshape(np.array(arr[0])[:,0],[-1,1])
        V = tf.reshape(np.array(arr[0])[:,1],[-1,1])
        a = tf.cast(tf.reshape(arr[1],[-1,1]),tf.float32)
        T = self.T
        dt = self.dt
        v = self.variables[network]
        t_i_plus_1 = T - t - dt
        b = self.b
        val = tf.cast(v[3]*t_i_plus_1+dt*(v[0]+v[1]*a+v[2]*(a**2)),tf.float32) 
        return tf.where(t<T, tf.pow(V, b) / b * val,0) 

    def get_all_variables(self):
        return self.variables['target'],self.variables['actual']
    
    def get_trainable_variables(self):
        return self.variables['actual']