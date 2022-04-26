import itertools

from torch.utils.tensorboard import SummaryWriter

from src.agent import DQNAgent
from src.environment import DQNEnvironment
import gym
from datetime import datetime


def deep_q_learning(environment_name, args):
    experiment_name = datetime.today().strftime('%Y-%m-%d') + '_' + environment_name

    writer = SummaryWriter()  # writes output to be consumed by TensorBoard

    environment = DQNEnvironment(environment=gym.make(environment_name), config=args)
    agent = DQNAgent(action_space=environment.action_space, config=args)

    evaluation_scores = []
    episode_done = 0

    step = 0  # no. of actions selected by the agent

    while step < args.n_training_steps:

        if episode_done:
            agent.clear_observation_buffer()
            environment.start_random_episode()

        # select actions using a uniform random policy until replay memory D is populated
        if not agent.memory.is_full():
            action = agent.action_space.sample()

        # otherwise, select action using an epsilon-greedy strategy
        else:
            action = agent.sample_action(step=step, mode='training')
            step += 1

        # execute action a_t in emulator
        observation, step_reward, episode_done, info = environment.step(action)

        # observe image x_(t+1)
        agent.observe(observation=observation)

        # store transition in replay memory D
        agent.store_experience(action, step, episode_done)

        # update the action-value function Q every k-th action
        if step % args.update_frequency == 0:
            loss = agent.update_network()
            writer.add_scalar('Loss', loss, step)

        # update target network every k-th update of the action-value function Q
        if step % (args.update_frequency * args.target_update_frequency) == 0:
            agent.update_target_network()

        if step % args.evaluation_frequency == 0:

            environment.start_episode()

            evaluation_episodes = 0
            evaluation_episode_reward = 0
            evaluation_reward = 0
            evaluation_step = 0

            while evaluation_step < args.n_evaluation_steps:
                evaluation_step += 1

                # select and execute action, observe image x_(t+1)
                action = agent.select_action(mode='evaluation')
                observation, step_reward, episode_done, _ = environment.step(action)
                agent.observe(observation=observation)

                # update evaluation reward
                evaluation_episode_reward += step_reward

                if episode_done and environment.get_lives():
                    evaluation_episodes += 1
                    evaluation_reward += evaluation_episode_reward
                    environment.start_episode()
                    writer.add_scalar('Evaluation Episode Reward', evaluation_episode_reward, step)

            average_evaluation_reward = evaluation_reward / max(1, evaluation_episodes)
            writer.add_scalar('Average Evaluation Reward', average_evaluation_reward, step)
            writer.add_scalar('Evaluation Episodes', evaluation_episodes, step)

            # store weights of network with the highest evaluation score
            if average_evaluation_reward > max(evaluation_scores):
                agent.save_model_weights(experiment_name)

            evaluation_scores.append(average_evaluation_reward)

    return max(evaluation_scores)
