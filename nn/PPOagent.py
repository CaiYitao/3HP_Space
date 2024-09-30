import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model.layer import GraphGPSLayer, HyperGraphLayer,CrossAttention
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GENConv, GINEConv,HypergraphConv
from torch_geometric.data.hypergraph_data import HyperGraphData
from torch.distributions import Categorical
from torch.cuda.amp import GradScaler, autocast
import copy
from collections import deque
import random
import wandb
import torch
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from buffer import choose_optimizer, choose_lr_scheduler, ReplayBuffer, Transition, Episode
import numpy as np  
from model.module import RuleActor, MolActor, HyperGraphModel



class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.rule_actor = RuleActor(config)
        self.mol_actor = MolActor(config)

    def forward(self, state):
        rule_prob = self.rule_actor(state)
        rule_idx = torch.argmax(rule_prob).item()
        mol_prob = self.mol_actor(state, rule_idx)
        return rule_prob, mol_prob

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.config = config
        self.hypergraph_model = HyperGraphModel(config)
        # self.rule_encoder = GraphEncoder(config)
        # self.rule_featurizer = RuleGraphFeaturizer()
        # self.mol_encoder = GraphEncoder(config)
        # # self.mol_featurizer = MolGraphFeaturizer()

        # self.cross_attention_state_rule = CrossAttention(config.dim_h, config.dim_h, config.dim_h)
        # self.cross_attention_state_mol = CrossAttention(config.dim_h, config.dim_h, config.dim_h)
        # self.cross_attention_rule_mol = CrossAttention(config.dim_h, config.dim_h, config.dim_h)
        self.mlp_v = nn.Sequential(
            nn.Linear(config.dim_h, config.dim_h),  # Concatenate state, rule, and mol representations
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_h, 1)  # Output Q-value
        )
     

    def forward(self, dg_state):
        # rule_rep = self.rule_encoder(self.rule_featurizer(rule))
        # mol_rep = self.mol_encoder(self.mol_featurizer(mol))
        state_rep = self.hypergraph_model(dg_state)

        
        # Pass through MLP to compute Q-value
        v = self.mlp_v(state_rep)
    
        return v
    



class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.actor = Actor(config)
        self.critic = Critic(config)

        self.apply(self._init_weights)
    
    def _init_weights(self,m):

        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, state):
        raise NotImplementedError  # Forward pass not needed for this class

    def act(self, state):
        rule_probs, mol_probs = self.actor(state)
        
        # Sample rule action
        rule_dist = Categorical(probs=rule_probs)
        rule_action = rule_dist.sample()
        rule_log_probs = rule_dist.log_prob(rule_action)
        
        # Sample molecule action
        mol_dist = Categorical(probs=mol_probs)
        mol_action = mol_dist.sample()
        mol_log_probs = mol_dist.log_prob(mol_action)

        return (rule_action, mol_action), (rule_log_probs, mol_log_probs)
    
    def evaluate(self, state, rule_action, mol_action):
        rule_probs, mol_probs = self.actor(state)
        
        # Evaluate rule action
        rule_dist = Categorical(probs=rule_probs)
        rule_log_probs = rule_dist.log_prob(rule_action)
        
        # Evaluate molecule action
        mol_dist = Categorical(probs=mol_probs)
        mol_log_probs = mol_dist.log_prob(mol_action)

        state_value = self.critic(state)

        # Calculate entropy
        rule_entropy = rule_dist.entropy()
        mol_entropy = mol_dist.entropy()

        return state_value, (rule_log_probs, mol_log_probs), (rule_entropy, mol_entropy)




class PPOAgent:
    def __init__(self, config, env):
        self.config = config
        self.policy = ActorCritic(config)
        self.optimizer = choose_optimizer(self.policy.parameters(), config.optimizer, lr=config.lr)
        self.scheduler = choose_lr_scheduler(self.optimizer, config.scheduler, **config.scheduler_params)
        self.buffer = ReplayBuffer(config)
        self.scaler = GradScaler()
        self.env = env

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))

    def collect_samples(self, num_episodes):
        self.buffer.reset()
        for _ in range(num_episodes):
            episode = Episode(self.config.discount)
            state = self.env.reset()  # Reset the environment
            for _ in range(self.config.max_steps):
                with torch.no_grad():
                    (rule_action, mol_action), (rule_log_probs, mol_log_probs) = self.policy.act(state)
                    wandb.log({"Rule Action": rule_action.item(), "Molecule Action": mol_action.item()})
                next_state, reward, done = self.env.step(rule_action.item(), mol_action.item())  # Perform action in the environment
                episode.append(Transition(state, (rule_action, mol_action), reward, (rule_log_probs, mol_log_probs)))
                state = next_state
                if done:
                    break
            self.buffer.add(episode)
            episode.reset()
        self.buffer.update_stats()

    def update(self):
        for _ in range(self.config.num_updates):
            transitions = self.buffer.sample()
            self._update_policy(transitions)
            self.scheduler.step()

    def _update_policy(self, transitions):

        with autocast():
            states_values,rule_log_probs,mol_log_probs,rule_entropy,mol_entropy = [],[],[],[],[]
            old_mol_log_probs,old_rule_log_probs = [],[]
            
            for transition in transitions:
                state_value, (rule_log_prob, mol_log_prob), (rule_ent, mol_ent) = self.policy.evaluate(transition.state, transition.action[0], transition.action[1])
                states_values.append(state_value)
                rule_log_probs.append(rule_log_prob)
                mol_log_probs.append(mol_log_prob)
                rule_entropy.append(rule_ent)
                mol_entropy.append(mol_ent)
                old_rule_log_probs.append(transition.log_probs[0])
                old_mol_log_probs.append(transition.log_probs[1])


            state_values = torch.stack(states_values)
            rule_log_probs = torch.stack(rule_log_probs)
            mol_log_probs = torch.stack(mol_log_probs)
            rule_entropy = torch.stack(rule_entropy)
            mol_entropy = torch.stack(mol_entropy)
            old_rule_log_probs = torch.stack(old_rule_log_probs)
            old_mol_log_probs = torch.stack(old_mol_log_probs)

            # Compute advantages
            advantages = (state_values - torch.tensor([t.g_return for t in transitions])).detach()

            # print(f"state values shape: {state_values.shape}")
            # print(f"advantages shape: {advantages.shape}")
            rule_ratio = torch.exp(rule_log_probs - old_rule_log_probs)
            clipped_rule_ratio = torch.clamp(rule_ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
            rule_policy_loss = -torch.min(rule_ratio * advantages, clipped_rule_ratio * advantages).mean()

            mol_ratio = torch.exp(mol_log_probs - old_mol_log_probs)
            clipped_mol_ratio = torch.clamp(mol_ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
            mol_policy_loss = -torch.min(mol_ratio * advantages, clipped_mol_ratio * advantages).mean()

            value_loss = F.mse_loss(state_values, torch.tensor([t.g_return for t in transitions]).unsqueeze(1))

            entropy_loss = -(rule_entropy.mean()+mol_entropy.mean())  # Entropy loss
            policy_loss = rule_policy_loss + mol_policy_loss
            total_loss = policy_loss + self.config.value_coeff * value_loss + self.config.entropy_coeff * entropy_loss
            print(f"total loss: {total_loss}")
            wandb.log({"Policy Loss": policy_loss.item(), "Value Loss": value_loss.item(), "Entropy Loss": entropy_loss.item(), "Total Loss": total_loss.item()})
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()


class PPORunner:
    def __init__(self, agent, config):
        self.agent = agent
        self.num_episodes = config.num_episodes
        self.log_interval = config.log_interval

    def train(self):
        for i in range(1, self.num_episodes + 1):
                
            self.agent.collect_samples(1)
            self.agent.update()
            if i % self.log_interval == 0:
                self.log_values()
                print(f"Episode {i}/{self.num_episodes}")
            if i == 200:
                self.agent.env.dump()

    def log_values(self):
        
        episode_rewards = self.agent.buffer.all_returns[-self.log_interval:]
        print(f"Episode Rewards: {episode_rewards}")
        mean_return = np.mean(episode_rewards)
        reward_variance = np.var(episode_rewards)  
        # mean_episode_length = np.mean(self.agent.buffer.all_episode_lengths[-self.log_interval:]) 

        wandb.log({
            "Episode Rewards": episode_rewards,
            "Mean Return": mean_return,
            "Reward Variance": reward_variance,
            # "Mean Episode Length": mean_episode_length,
        })
