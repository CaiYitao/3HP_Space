from buffer import ReplayBuffer, Episode
import wandb

class Runner:
    def __init__(self, agent, env, replay_buffer, config):
        self.agent = agent
        self.env = env
        self.replay_buffer = replay_buffer
        self.max_episodes = config.max_episodes
        self.max_steps_per_episode = config.max_steps_per_episode
        self.batch_size = config.batch_size
        self.training_epochs = config.epochs
        self.discount_factor = config.discount_factor

    def run(self):
        for episode_num in range(self.max_episodes):
            episode = Episode(discount_factor=self.discount_factor)
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.max_steps_per_episode):
                action_rule, action_molecule = self.agent.select_action(state)
                
                next_state, reward, done, _ = self.env.step(action_rule, action_molecule)
                episode.add_transition(state, action_rule, action_molecule, reward)
                
                self.replay_buffer.add(state, action_rule, action_molecule, next_state, reward, done)
                state = next_state
                episode_reward += reward

                if len(self.replay_buffer) >= self.batch_size:
                    self.agent.train(self.replay_buffer, self.batch_size, self.training_epochs)

                if done:
                    break

            episode.calculate_returns()
            # wandb.log({"Episode Reward": episode_reward, "Episode Returns": episode.returns[0]}, step=episode_num)

            print(f"Episode {episode_num + 1}/{self.max_episodes}, Reward: {episode_reward}, Returns: {episode.returns[0]}")



        