# Replication of "Human-Level Control Through Deep Reinforcement Learning"
[![](https://img.shields.io/github/license/jannik-brinkmann/dqn.svg)](https://github.com/jannik-brinkmann/dqn/blob/master/LICENSE.md)

### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Implementation

This repository contains a replication of [Human-Level Control Through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236) using OpenAI Gym and PyTorch. The implementation includes an environment wrapper designed to be applicable to all environments, although it has only been evaluated on Breakout and Pong so far.

### Evaluation

The agent achieved a game score of 129.8 on Breakout, compared to the original authors' score of 316.6. In general, as with many deep reinforcement learning methods, reproducing results can be challenging due to non-determinism in standard benchmark environments and intrinsic variance in the methods themselves [(Henderson et al. 2018)](https://ojs.aaai.org/index.php/AAAI/article/view/11694). However, it's possible that a bug in the code contributes to this effect.

<div align="center">
  <video src="https://user-images.githubusercontent.com/62884101/226115786-cae633c3-9ba8-4952-8e9f-fedbc6267cff.mp4" width="100" />
</div>

### Citation   
```
@article{
  title={Human-Level Control Through Deep Reinforcement Learning},
  author={Volodymyr Mnih and Koray Kavukcuoglu and David Silver and Andrei A. Rusu and Joel Veness and Marc G. Bellemare and Alex Graves and Martin Riedmiller and Andreas K. Fidjeland and Georg Ostrovski and Stig Petersen and Charles Beattie and Amir Sadik and Ioannis Antonoglou and Helen King and Dharshan Kumaran and Daan Wierstra and Shane Legg and Demis Hassabis},
  journal={Nature},
  year={2015}
}
```
