# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
"""
    A sample actor network. Please see other actor networks defined in the folder for some other flavors of the actor network.
"""
class A:
    def __init__(self,cfg):
        """
        Intializes the actor network. All the variables are very self explanatory in the initialization.

        Parameters:
            cfg - The config of the experiment. 
        """
        self.m=cfg['ddpg']['a']['variables']['m']
        #self.s=cfg['ddpg']['a']['variables']['s']
        self.variables ={}
        self.variables['actual'] = [tf.Variable(self.m,dtype="float32")]
        self.variables['target'] =[self.m]
        self.cfg = cfg
        self.r = self.cfg['env']['r']
        self.T = self.cfg['env']['T']
        self.dt = self.cfg['env']['dt']
        self.update_actor = True
        
        
    def mu(self,X,network= 'actual'):
        """
            The optimal action finding network. This has to be implemented in any new Actor file. 

            Parameters:
                X - The state of the environment. Typically the (wealth,time) tuple in our experiments.
                network - The variable that is either 'actual' or 'target' depending on the network.
        """
        m = self.variables[network][0]
        #s = self.variables[network][1]
        val = m #(m-self.r)/(s**2*(1-self.b))
        a = tf.where(X[...,0]< self.T,val,0)
        return a
        
    def custom_update(self,ddpg_agent):
        """"
            Customizing the update step. If we do not need the DDPG to update the network, and have it done in a non standard way, then this method has to be implemented. This is used in conjunction with the self.update_actor variable which has to be set to False for this function to be called from the DDPG update method.

            Parameters:
                ddpg_agent:  The DDPG agent itself.
        """
        for i in range(len(self.variables['actual'])):
            self.variables['actual'][i]=ddpg_agent.qN.variables['actual'][i]
            self.variables['target'][i]=ddpg_agent.qN.variables['target'][i]

    def get_all_variables(self):
        """
            This function has to be defined for any new Actor network and this should list all the variables in the network , both that can be trained and not. Typically target variables are not trained.
        """
        return self.variables['target'],self.variables['actual']
    
    def get_trainable_variables(self):
        """
            This function has to be defined for any new Actor network and this should list all the trainable variables in the network Typically 'actual' variables are trained.
        """
        return self.variables['actual']