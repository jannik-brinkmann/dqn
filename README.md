# Replication of "Human-Level Control Through Deep Reinforcement Learning"
[![](https://img.shields.io/github/license/jannik-brinkmann/dqn.svg)](https://github.com/jannik-brinkmann/dqn/blob/master/LICENSE.md)

### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Training
To train the Deep Q-Network use the script using this command:

```bash
python run_image_classification.py \
    --dataset_name beans \
    --output_dir ./beans_outputs/ \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --push_to_hub \
    --push_to_hub_model_id vit-base-beans \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337
```

### Implementation

This repository contains a replication of [Human-Level Control Through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236) using OpenAI Gym and PyTorch. The implementation includes an environment wrapper designed to be applicable to all environments, although it has only been evaluated on Breakout and Pong so far.

### Evaluation

The agent achieves a game score of 129.8 on Breakout, compared to the original authors' score of 316.6. In general, as with most deep reinforcement learning methods, reproducing results can be challenging due to non-determinism in standard benchmark environments and intrinsic variance in the methods themselves [(Henderson et al. 2018)](https://ojs.aaai.org/index.php/AAAI/article/view/11694). However, it's possible that a bug in the code contributes to this effect.

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
