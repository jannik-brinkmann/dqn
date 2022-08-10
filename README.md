# Deep Q-Network

A PyTorch re-implementation of ["Human-Level Control Through Deep Reinforcement Learning"](https://www.nature.com/articles/nature14236). 

#### Implementation Details
The environment wrapper has been implemented in an attempt to be applicable to all OpenAI Gym Atari environments. However, it has not been evaluated on environments other than Breakout and Pong, wherefore specialties of other environments might not be considered. 

#### Training Details

In general, results of deep reinforcement learning methods are difficult to reproduce due to non-determinism in standard benchmark envrionments, combined with variance intrinsic to the methods [(Henderson, Peter, et al.)](https://ojs.aaai.org/index.php/AAAI/article/view/11694).
