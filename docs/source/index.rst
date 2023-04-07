.. rl-fn documentation master file, created by
   sphinx-quickstart on Sun Apr  2 19:13:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rl-fn's documentation!
=================================
**rl-fn** is A DDPG based solution for portfolio optimization that uses specific function classes for actor and critic parts of DDPG instead of neural networks. There are also customized implementations of DDPG (DDPGShockBufferand DDPGShockBufferEstimate)  that improve accuracy as they build better estimates of the expected Q values  of future states. This is a configurable framework where any combination of Q and A can be plugged and played (defined in the folders Qn and An). The environment also can be redefined or configured by editing the BSAvgState file. There is a tunable configuration to experiment with different classes of experiments. Finally there are utilites to plot, visualize the results

.. note::

   This project is under maintanence.
   
   .qN
   .aN
   .

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   usage
   api
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`