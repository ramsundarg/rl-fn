# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
class A:
    def __init__(self,cfg):
        
        self.m=cfg['ddpg']['a']['variables']['m']
        self.s=cfg['ddpg']['a']['variables']['s']
        self.variables ={}
        self.variables['actual'] = [tf.Variable(self.m,dtype="float32"),tf.Variable(self.s,dtype="float32")]
        self.variables['target'] =[self.m,self.s]
        self.cfg = cfg
        self.r = self.cfg['env']['r']
        self.b = self.cfg['env']['b']
        self.T = self.cfg['env']['T']
        self.dt = self.cfg['env']['dt']
        self.update_actor = False
        
        
    def mu(self,X,network= 'actual'):
        t = tf.reshape(np.array(X)[:,0],[-1,1])
        V = tf.reshape(np.array(X)[:,1],[-1,1])
        m = self.variables[network][0]
        s = self.variables[network][1]
        val = (m-self.r)/(s**2*(1-self.b))
        a = tf.where(X[:,0]< self.T,val,0)
        return a
        
    def custom_update(self,ddpg_agent):
        for i in range(len(self.variables['actual'])):
            self.variables['actual'][i]=ddpg_agent.qN.variables['actual'][i]
            self.variables['target'][i]=ddpg_agent.qN.variables['target'][i]

    def get_all_variables(self):
        return self.variables['target'],self.variables['actual']
    
    def get_trainable_variables(self):
        return self.variables['actual']