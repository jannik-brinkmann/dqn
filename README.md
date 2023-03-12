# Deep Q-Network

A PyTorch implementation of ["Human-Level Control Through Deep Reinforcement Learning"](https://www.nature.com/articles/nature14236). 

#### Implementation Details
I implemented the environment wrapper in an attempt to make it applicable to all OpenAI Gym Atari environments. However, I have not evaluated it on environments other than Breakout and Pong, wherefore specifics of other environments might not be considered. 

#### Training Details
I use the training parameters as described in the original paper, except I use a smaller replay memory (100k) and start experience replay earlier (after 25k steps). 

#### Evaluation
In general, results of deep reinforcement learning methods are difficult to reproduce due to non-determinism in standard benchmark envrionments, combined with variance intrinsic to the methods [(Henderson, Peter, et al.)](https://ojs.aaai.org/index.php/AAAI/article/view/11694).
