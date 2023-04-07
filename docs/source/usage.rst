.. rl-fn documentation master file, created by
   sphinx-quickstart on Sun Apr  2 19:13:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Usage
=================================


The most common way of running the project is to use the following command.

.. code-block:: console

   python run_all.py cfg/<cfg.json>

Describing an experiment in cfg.json
-------------------------------------

An experiment can be described completely in a cfg file.  A sample is described below


.. code-block:: json

   {
      "env": {
         "name": "BSEnv",
         "mu": 0.16,
         "sigma": 0.8,
         "r": 0.0,
         "T": 2,
         "dt": 0.5,
         "V_0": 0.5,
         "U_2" :"utility_functions.power_utility"
      },
      "general_settings": {
         "max_episodes": 5000,
         "max_steps": 100,
         "batch_size": 2048
      },
      "ddpg": {
         "gamma": 1,
         "tau": 0.01,
         "buffer_len": 100000,
         "q": {
            "name": "q_pow_ut",
            "lr": 0.001,
            "variables": {
            "m": 0.5,
            "s": 0.5
            }
         },
         "a": {
            "name": "a_pow_ut1",
            "lr": 0.001,
            "variables": {
            "m": 0.1
            
            }
         }
      },
      "b": 0.5



   }

- *env* - Refers to the enviroment parameters that can be set.

   - *name* - Name of the python module that controls the environment.  Implement your own environment or add more functions to BSAvgState. BSAvgState is a black scholes enviroment and has functions that are utilized by the 3 versions of DDPG. If you need to implement a new enviroment, the most basic functions that have to be implemented are
      - *reset* ()

      - *step* () should return the state , reward, next state and a dictionary tuple that is used by run_experiment trainer function

   Completely define the variables that the enviroment will need (in the black scholes case, we needed model parameters mu and sigma). Some useful variables are dt,T,v_0 and u_2, which are generally needed to describe states in any RL framework. They are not a must by design but required in most cases.

- *general_settings* -  The variables that can be defined are quite self explanatory. Define these variables based on the experiment that is run. A neural network based DDPG might require more iterations than the one described above.

- *ddpg* - 
   - gamma - The discounted reward factor that is set to 1
   - tau - The target parameter learning rate
   - buffer_len - The replay buffer batch_size

- There are 2 components, the **Q component that is the critic component** and the **actor component which is defined as a**. The Q modules must be defined in qN and the actor components must be defined in aN folders respectively.  There are some samples that are already defined. They can also be reused.

   - Q component: 
      While defining a new Q component, the following function must be defined.

      .. code-block:: python

         def __init__(self,cfg):
               
         def q_mu(self,arr,network='actual'):
        
         def get_all_variables(self):

         def get_trainable_variables(self,arr,network='actual'):

      q_mu is the actual function that returns the Q-value of state,action pair. Look at the examples provided in the sample q files to look at what the variables should return.

   - A component: 
      It is quite similar to the Q component, the madatory functions that have to be defined are

      .. code-block:: python

         def __init__(self,cfg):
               
         def mu(self,arr,network='actual'):
        
         def get_all_variables(self):

         def get_trainable_variables(self,arr,network='actual'):

      There is also a piece where you can customize if you need to custom update the actor component on every training step (if you choose to override the default behavior of DDPG that would apply the gradients to the trainable variables). To enable this you need to define a separate function

      .. code-block:: python

         def custom_update(self,ddpg_agent):
      
      and to activate this mode, you need to set a special variable self.update_actor as True. Then on each update step, DDPG will call the custom_update function instead of the usual behavior.


DDPG module
--------------

All DDPG implementations derive from a common class called CommonDDPG.py. There is the default implementation DDPG.py which covers the base case. There are some customized versions, the details of which can be found in the Thesis document in the attached project. They are defined as DDPGShockBuffer and DDPGShockBufferEstimate. In practice, choose the flavor of DDPG for the specific problem.

The main functions to be re-defined while implementing your own DDPG flavor are the following. Explaining it via the DDPG implmentation method.

.. code-block:: python

   class DDPG(CommonDDPG.DDPG):
    def __init__(self,cfg):

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):

    def update(
        self,attr_dict
        
    ):

    # We compute the loss and update parameters
    def learn(self,episode):
       

One should initialize the replay buffer (described in the next section) in the init function. 

- The record function gets a tuple as recorded in the env module (which can record any parameter besides the important ones state,reward, next state).  The DDPG itself can record other internal parameters along with it. All this can be stored in the replay buffer by calling the method record. The record method the replay buffer needs a dictionary with all these parameters that are stored as keys and their corresponding values as values of those keys in the dictionary. So there is complete flexibility in what can be stored as the replay buffer does not specifically search for any particular parameter.


- The update function is the main function of DDPG that is used to update gradients of the actor and critic components.  The attr_dict is the mini-batch and it is a dictionary with parameters and the mini batch samples of them. Look at the default definition in DDPG.py. It gives a guideline on how the gradients should be updated. Besides there are also hooks in it to call the custom actor update function defined in the previous section.

- The learn method is called from the run_experiment method on every training step. The method should take a mini bacth of parameters it is interested from the replay buffer. Then it must call the learn function of the base method (see the DDPG definition) which will interally call the update method defined in this class and also update the target networks as specified in the configuration.  Again there are hooks in the CommonDDPG, that will pass on the update weight of the target network to Q or A component even. The goal is to provide flexibility to almost all the steps in DDPG while having reusable components defined in the common classes.






















