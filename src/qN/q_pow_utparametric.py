
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
class Q:
    """
        A sample critic network that works on the parametric form of the actual function and computes the parameters by DDPG. Please see other critic networks defined in the folder for some other flavors of the actor network.
    """    
    def __init__(self,cfg):
        """
        Intializes the actor network. All the variables are very self explanatory in the initialization.

        Parameters:

            cfg - The config of the experiment. 
        """
        self.v=cfg['ddpg']['q']['variables']
        
        self.variables ={}
        self.variables['actual'] = [tf.Variable(self.v[0],dtype="float32"),tf.Variable(self.v[1],dtype="float32"),tf.Variable(self.v[2],dtype="float32"),tf.Variable(self.v[3],dtype="float32")]
        self.variables['target'] =[self.v[0],self.v[1],self.v[2],self.v[3]]
        self.cfg = cfg
        self.b = cfg['env']['b']
        self.T = self.cfg['env']['T']
        self.dt = self.cfg['env']['dt']
        
        
    def q_mu(self,arr,network='actual'):
        """
            The Q value finding function. This has to be implemented in any new Critic file. 

            Parameters:

                arr - The state of the environment along with the polled action for that state. Typically the (wealth,time) tuple is the state in our experiments.

                network - The variable that is either 'actual' or 'target' depending on the network.

            Returns:
            
                The actual Q value for the mini-batch
        """
        t = tf.cast(tf.reshape(arr[0][...,0],[-1,1]),tf.float32)
        V = tf.cast(tf.reshape(arr[0][...,1],[-1,1]),tf.float32)
        a = tf.cast(tf.reshape(arr[1],[-1,1]),tf.float32)
        T = self.T
        dt = self.dt
        v = self.variables[network]
        t_i_plus_1 = T - t - dt
        b = self.b
        val = tf.cast(v[3]*t_i_plus_1+dt*(v[0]+v[1]*a+v[2]*(a**2)),tf.float32) 
        return tf.where(t<T, tf.pow(V, b) / b * val,0) 

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