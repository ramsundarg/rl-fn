
import tensorflow as tf
import numpy as np
class Q:
    def __init__(self,cfg):
        s_input_critic = tf.keras.layers.Input(shape=(2, ))
        a_input_critic = tf.keras.layers.Input(shape=(1, ))
        last_critic = tf.keras.layers.Concatenate()([s_input_critic, a_input_critic])
        last_critic = tf.keras.layers.Dense(32,activation='tanh')(last_critic)
        last_critic = tf.keras.layers.Dense(32,activation='tanh')(last_critic)
        output_critic = tf.keras.layers.Dense(1,activation=None)(last_critic)
        q_model = tf.keras.Model([s_input_critic, a_input_critic], output_critic)

        s_input_critic_tg = tf.keras.layers.Input(shape=(4, ))
        a_input_critic_tg = tf.keras.layers.Input(shape=(1, ))
        last_critic_tg = tf.keras.layers.Concatenate()([s_input_critic_tg, a_input_critic_tg])
        last_critic_tg = tf.keras.layers.Dense(32,activation='tanh')(last_critic_tg)
        last_critic_tg = tf.keras.layers.Dense(32,activation='tanh')(last_critic_tg)
        output_critic_tg = tf.keras.layers.Dense(1,activation=None)(last_critic_tg)
        q_tg_model = tf.keras.Model([s_input_critic_tg, a_input_critic_tg], output_critic_tg)
        
        self.m=cfg['ddpg']['q']['variables']['m']
        self.s=cfg['ddpg']['q']['variables']['s']
        self.variables ={}
        self.variables['actual'] = [tf.Variable(self.m,dtype="float32"),tf.Variable(self.s,dtype="float32")]
        self.variables['target'] =[self.m,self.s]
        self.cfg = cfg
        self.r = self.cfg['env']['r']
        self.T = self.cfg['env']['T']
        self.dt = self.cfg['env']['dt']
        
        
    def q_mu(self,arr,network='actual'):
        t = tf.reshape(np.array(arr[0])[:,0],[-1,1])
        V = tf.reshape(np.array(arr[0])[:,1],[-1,1])
        a = tf.cast(tf.reshape(arr[1],[-1,1]),tf.float32)
        T = self.T
        r= self.r
        dt = self.dt
        mu = self.variables[network][0]
        sigma = self.variables[network][1]
        r_ex = mu - r
        sigma_2 = sigma ** 2
        t_i_plus_1 = T - t - dt
        val = tf.cast(tf.math.log(V) + (r + r_ex*a- 0.5*sigma_2*a**2)*dt +( r+ 0.5*((r_ex/sigma)**2))*t_i_plus_1,tf.float32)

        
        return tf.where(t<T,val,0)       

    def get_all_variables(self):
        return self.variables['target'],self.variables['actual']
    
    def get_trainable_variables(self):
        return self.variables['actual']
# origin Q net (Q_phi)


# target Q net (Q_phi')
s_input_critic_tg = tf.keras.layers.Input(shape=(4, ))
a_input_critic_tg = tf.keras.layers.Input(shape=(1, ))
last_critic_tg = tf.keras.layers.Concatenate()([s_input_critic_tg, a_input_critic_tg])
last_critic_tg = tf.keras.layers.Dense(64,activation='relu')(last_critic_tg)
output_critic_tg = tf.keras.layers.Dense(1,activation=None)(last_critic_tg)
q_tg_model = tf.keras.Model([s_input_critic_tg, a_input_critic_tg], output_critic_tg)