import itertools

from torch.utils.tensorboard import SummaryWriter

from src.agent import DQNAgent
from src.environment import DQNEnvironment
import gym
from datetime import datetime


def deep_q_learning(environment_name, experiment_name, args):

    writer = SummaryWriter()  # writes output to be consumed by TensorBoard

    environment = DQNEnvironment(environment=gym.make(environment_name), config=args)
    agent = DQNAgent(action_space=environment.action_space, config=args)

    episode_done = True
    episode_reward = 0
    evaluation_scores = []

    step = 1  # no. of actions selected by the agent
    while step < args.n_training_steps:

        if episode_done:
            writer.add_scalar('Episode Reward', episode_reward, step)
            episode_reward = 0
            agent.clear_observation_buffer()
            environment.start_random_episode()

        # sample actions using a uniform random policy until replay memory D is populated
        if not agent.memory.is_full():
            action = agent.action_space.sample()

        # otherwise, select action using an epsilon-greedy strategy
        else:
            action = agent.sample_action(step=step, mode='training')
            step += 1

        # execute action a_t, observe image x_(t+1), store transition in replay memory D
        observation, step_reward, episode_done, info = environment.step(action)
        agent.observe(observation=observation)
        agent.store_experience(action, step_reward, episode_done)

        episode_reward += step_reward

        # update the action-value function Q every k-th action
        if step % args.update_frequency == 0:
            loss = agent.update_network()
            writer.add_scalar('Loss', loss, step)

        # update target network every k-th update of the action-value function Q
        if step % (args.update_frequency * args.target_update_frequency) == 0:
            agent.update_target_network()

        # evaluate agent performance every k-th action
        if step % args.evaluation_frequency == 0:
            average_evaluation_reward_per_episode = evaluate_agent(agent, environment, args.n_evaluation_steps)
            writer.add_scalar('Avg. Evaluation Reward per Episode', average_evaluation_reward_per_episode, step)

            # store weights of network with the highest evaluation score
            if average_evaluation_reward_per_episode > max(evaluation_scores):
                agent.save_model_weights(experiment_name)

            evaluation_scores.append(average_evaluation_reward_per_episode)

    return max(evaluation_scores)


def evaluate_agent(agent, environment, n_evaluation_steps):

    environment.start_episode()

    # evaluate agent as average reward over evaluation episodes
    reward = 0
    episode_reward = 0

    episodes = 0
    step = 0
    while step < n_evaluation_steps:
        step += 1

        # select and execute action, observe image x_(t+1)
        action = agent.select_action(mode='evaluation')
        observation, step_reward, done, _ = environment.step(action)
        agent.observe(observation=observation)

        episode_reward += step_reward

        if done:
            environment.start_episode()

            # in evaluation, episodes don't end when losing a life
            if environment.is_game_over():
                episodes += 1
                reward += episode_reward
                episode_reward = 0

    average_reward_per_episode = reward / max(1, episodes)
    return average_reward_per_episode
