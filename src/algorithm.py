import itertools

from torch.utils.tensorboard import SummaryWriter


def deep_q_learning(agent, environment, config):

    game_reward = 0.0  # one game might consist of multiple episode (e.g. in Breakout)
    n_steps = 1  # no. of actions selected by the agent
    writer = SummaryWriter()  # writes output to be consumed by TensorBoard

    for episode in itertools.count():

        if config.mode == 'training':
            environment.start_random_episode()
        else:
            environment.start_episode()

        agent.clear_observation_buffer()

        episode_done = False
        episode_reward = 0.0

        while not episode_done:

            # in training mode, select actions using a uniform random policy until replay memory D is populated
            if config.mode == 'training' and not agent.memory.is_full():
                action = agent.action_space.sample()

            # otherwise, select action using an epsilon-greedy strategy
            else:
                action = agent.select_action(n_steps=n_steps)
                n_steps += 1

            # execute action a_t in emulator
            observation, step_reward, episode_done, info = environment.step(action)

            # observe image x_(t+1); store transition in replay memory D
            agent.make_observation(action, observation, step_reward, episode_done)

            # update the action-value function Q every k-th action
            if config.mode == 'training' and n_steps % config.update_frequency == 0:
                loss = agent.update_network()
                writer.add_scalar('Loss', loss, n_steps)

            # update target network every k-th update of the action-value function Q
            if config.mode == 'training' and n_steps % (config.update_frequency * config.target_update_frequency) == 0:
                agent.save_model_weights('checkpoint_' + str(n_steps) + '.pth.tar')
                agent.update_target_network()

            episode_reward += step_reward

        writer.add_scalar('Episode Reward', episode_reward, episode)

        game_reward += episode_reward
        if environment.is_game_over():
            writer.add_scalar('Game Reward', game_reward, episode)
            game_reward = 0.0
