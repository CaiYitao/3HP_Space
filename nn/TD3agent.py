import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model.layer import GraphGPSLayer, HyperGraphLayer,CrossAttention
from CRN_IMA.featurizer import HyperGraphFeaturizer, RuleGraphFeaturizer,MolGraphFeaturizer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GENConv, GINEConv,HypergraphConv
from torch_geometric.data.hypergraph_data import HyperGraphData
from torch.distributions import Categorical
from torch.cuda.amp import GradScaler, autocast
import copy
from collections import deque
import random
from buffer import ReplayBuffer,choose_optimizer,choose_lr_scheduler
import wandb
from module import HyperGraphModel, GraphEncoder, RuleActor, MolActor


# class HyperGraphModel(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.config = cfg
#         self.encoder = GraphEncoder(cfg)
#         self.hypergraph_featurizer = HyperGraphFeaturizer()
#         self.hypergraph_layer = HypergraphConv(cfg.dim_h,cfg.dim_h)
#         self.mlp = nn.Sequential(nn.Linear(cfg.dim_h, cfg.dim_h*2),
#                                 nn.GELU(),
#                                 nn.Dropout(cfg.dropout),
#                                 nn.Linear(cfg.dim_h*2, cfg.dim_h))

#     def forward(self, dg):
#         hg_data = self.hypergraph_featurizer(dg)
#         if torch.cuda.is_available():
#             hg_data = hg_data.to(self.config.device)

#         print(f"hg_data.x: {hg_data.x}")
#         mol_loader = DataLoader(hg_data.x)
#         rule_loader = DataLoader(hg_data.edge_attr)
#         mol_rep = torch.empty(0, self.config.dim_h)
#         for i, mol_graph_data in enumerate(mol_loader):
#             output = self.encoder(mol_graph_data)
#             mol_rep = torch.stack([mol_rep, output], 0)
#         rule_rep = torch.empty(0, self.config.dim_h)
#         for i, rule_graph_data in enumerate(rule_loader):

#             rule_graph_batch = Batch.from_data_list(rule_graph_data)
#             output = self.encoder(rule_graph_batch)
#             rule_rep = torch.stack([rule_rep, output], 0)
        
#         hypergraph_data = HyperGraphData(x=mol_rep, edge_index=hg_data.edge_index, edge_attr=rule_rep)

#         outputs = []
#         for i in range(self.config.num_hglayers):
#             hypergraph_data = self.hypergraph_layer(hypergraph_data)
#             outputs.append(hypergraph_data.x)
#         if self.config.parallel_output:
#             output = torch.sum(torch.stack(outputs), 0)
#             return self.mlp(output)
#         else:
#             return self.mlp(hypergraph_data.x)


# class GraphEncoder(nn.Module):
#     def __init__(self, cfg):
#         super(GraphEncoder, self).__init__()
#         self.config = cfg
#         self.layer = GraphGPSLayer(cfg)
#         self.mlp = nn.Sequential(nn.Linear(cfg.dim_h, cfg.dim_h*2),
#                                 nn.GELU(),
#                                 nn.Dropout(cfg.dropout),
#                                 nn.Linear(cfg.dim_h*2, cfg.dim_h))
    
#     def forward(self, batch):
#         outputs = []
#         for i in range(self.config.num_layers):
#             batch = self.layer(batch)
#             outputs.append(batch.x)
#         if self.config.parallel_output:
#             output = torch.sum(torch.stack(outputs), 0)
#             return torch.sum(self.mlp(output), 0)
#         else: 
#             return torch.sum(self.mlp(batch.x),0)



# class RuleActor(nn.Module):
#     def __init__(self, config):
#         super(RuleActor, self).__init__()
#         self.config = config
#         self.hypergraph_model = HyperGraphModel(config)
#         self.mlp = nn.Sequential(nn.Linear(config.dim_h, config.dim_h*2),
#                                         nn.GELU(),
#                                         nn.Dropout(config.dropout),
#                                         nn.Linear(config.dim_h*2, config.dim_h),
#                                         nn.GELU(),
#                                         nn.Dropout(config.dropout),
#                                         nn.Linear(config.dim_h, config.rule_dim))


#     def forward(self,dg_state):
#         rep = self.hypergraph_model(dg_state)
#         return torch.softmax(self.mlp(rep), 0)


# class MolActor(nn.Module):
#     rule_idx_dict = {0: "R1: (S)-lactate + NAD+ = pyruvate + NADH + H+",
#                     1:"R2: Ac-aldehyde + CoA + NAD+ = Ac-CoA + NADH + H+",
#                     2:"R3: L-Aspartate + 2-Oxoglutarate <=> Oxaloacetate + L-Glutamate",
#                     3:"R4: Acyl-CoA + H2O <=> CoA + Carboxylate",
#                     4:"R5: SAM + CO2 = SAM-CO2H",
#                     5:"R6: L-malate = fumarate + H2O",
#                     6:"R7: Lactoyl-CoA <=> Propenoyl-CoA + H2O",
#                     7:"R8: L-Aspartate <=> Fumarate + Ammonia",
#                     8:"R9: L-lysine = (3S)-3,6-diaminohexanoate",
#                     9:"R10: ATP + Acetate + CoA <=> AMP + PPi + Ac-CoA",
#                     10:"R11: ATP + Ac-CoA + HCO3- = ADP + malonyl-CoA + Pi"}
#     def __init__(self, config):
#         super(MolActor, self).__init__()
#         self.config = config
#         self.hypergraph_model = HyperGraphModel(config)
#         self.rule_encoder = GraphEncoder(config)
#         self.rule_featurizer = RuleGraphFeaturizer()
#         self.cross_attention = CrossAttention(config.dim_h, config.dim_h, config.dim_h)
#         self.mlp = nn.Sequential(nn.Linear(config.dim_h, config.dim_h*2),
#                                         nn.GELU(),
#                                         nn.Dropout(config.dropout),
#                                         nn.Linear(config.dim_h*2, config.dim_h),
#                                         nn.GELU(),
#                                         nn.Dropout(config.dropout),
#                                         nn.Linear(config.dim_h, config.mol_dim))


#     def forward(self,dg_state,rule):
#         rule_name = self.rule_idx_dict[rule]
#         rule_rep = self.rule_encoder(self.rule_featurizer(rule_name))
#         state_rep = self.hypergraph_model(dg_state)
#         attended_output = self.cross_attention(state_rep, rule_rep)
#         output = self.mlp(attended_output)
#         return torch.softmax(output, dim = -1)  

class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.rule_actor = RuleActor(config)
        self.mol_actor = MolActor(config)

    def forward(self, state):
        rule = self.rule_actor(state)
        mol = self.mol_actor(state, rule)
        return rule, mol

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.config = config
        self.hypergraph_model = HyperGraphModel(config)
        self.rule_encoder = GraphEncoder(config)
        self.rule_featurizer = RuleGraphFeaturizer()
        self.mol_encoder = GraphEncoder(config)
        self.mol_featurizer = MolGraphFeaturizer()
        self.cross_attention_state_rule = CrossAttention(config.dim_h, config.dim_h, config.dim_h)
        self.cross_attention_state_mol = CrossAttention(config.dim_h, config.dim_h, config.dim_h)
        self.cross_attention_rule_mol = CrossAttention(config.dim_h, config.dim_h, config.dim_h)
        self.mlp_Q1 = nn.Sequential(
            nn.Linear(config.dim_h * 3, config.dim_h),  # Concatenate state, rule, and mol representations
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_h, 1)  # Output Q-value
        )
        self.mlp_Q2 = nn.Sequential(
            nn.Linear(config.dim_h * 3, config.dim_h),  # Concatenate state, rule, and mol representations
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_h, 1)  # Output Q-value
        )

    def forward(self, dg_state, rule,mol):
        rule_rep = self.rule_encoder(self.rule_featurizer(rule))
        mol_rep = self.mol_encoder(self.mol_featurizer(mol))
        state_rep = self.hypergraph_model(dg_state)
        if self.config.cross_attention:
            state_rule_rep = self.cross_attention_state_rule(state_rep, rule_rep)
            state_mol_rep = self.cross_attention_state_mol(state_rep, mol_rep)
            rule_mol_rep = self.cross_attention_rule_mol(rule_rep, mol_rep)
            combined_rep = torch.cat((state_rule_rep, state_mol_rep, rule_mol_rep), dim=1)
        
        # # Concatenate state, rule, and mol representations
        else: 
            combined_rep = torch.cat((state_rep, rule_rep, mol_rep), dim=1)
        
        # Pass through MLP to compute Q-value
        Q1 = self.mlp_Q1(combined_rep)
        Q2 = self.mlp_Q2(combined_rep)
        return Q1, Q2
    
    def Q1(self, dg_state, rule, mol):
        rule_rep = self.rule_encoder(self.rule_featurizer(rule))
        mol_rep = self.mol_encoder(self.mol_featurizer(mol))
        state_rep = self.hypergraph_model(dg_state)
        if self.config.cross_attention:
            state_rule_rep = self.cross_attention_state_rule(state_rep, rule_rep)
            state_mol_rep = self.cross_attention_state_mol(state_rep, mol_rep)
            rule_mol_rep = self.cross_attention_rule_mol(rule_rep, mol_rep)
            combined_rep = torch.cat((state_rule_rep, state_mol_rep, rule_mol_rep), dim=1)
        
        # # Concatenate state, rule, and mol representations
        else: 
            combined_rep = torch.cat((state_rep, rule_rep, mol_rep), dim=1)
        
        # Pass through MLP to compute Q-value
        Q1 = self.mlp_Q1(combined_rep)

        return Q1

class TD3Agent:
    def __init__(self, config):
        self.config = config

        self.actor = Actor(config)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = choose_optimizer(self.actor.parameters(), config.actor_optimizer, **config.actor_optimizer_params)

        self.critic = Critic(config)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = choose_optimizer(self.critic.parameters(), config.critic_optimizer, **config.critic_optimizer_params)
        self.target_update_interval = config.target_update_interval
        self.timestep = 0

        # Weights for the loss terms
        self.critic_loss_weight = config.critic_loss_weight
        self.actor_loss_weight = config.actor_loss_weight

        # Learning rate scheduler for actor and critic optimizers
        self.actor_scheduler = choose_lr_scheduler(self.actor_optimizer, config.actor_scheduler_type, **config.actor_scheduler_params)
        self.critic_scheduler = choose_lr_scheduler(self.critic_optimizer, config.critic_scheduler_type, **config.critic_scheduler_params)

        # Gradient clipping
        self.gradient_clip = config.gradient_clip
        self.max_action = config.max_action
        # GradScaler for mixed-precision training
        self.scaler = GradScaler()        # Initialization code as previously defined

    def train(self, replay_buffer):
        for epoch in range(self.config.n_epochs):
            # Sample a batch of transitions from the replay buffer
            state, action_rule, action_molecule, next_state, reward, done = replay_buffer.sample(self.config.batch_size)

            # Convert to appropriate tensor formats, assuming necessary preprocessing
            # For graph-based environments, ensure this process retains structural integrity
            
            with autocast(enabled=self.config.use_amp):
                # Compute target actions with noise
                noise = (torch.randn_like(action_rule) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action_rule, next_action_molecule = self.actor_target(next_state)
                # Ensure actions are within bounds
                next_action_rule = (next_action_rule + noise).clamp(-self.max_action, self.max_action)
                next_action_molecule = (next_action_molecule + noise).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action_rule, next_action_molecule)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + ((1.0 - done) * self.config.discount * target_Q).detach()

                # Get current Q estimates
                current_Q1, current_Q2 = self.critic(state, action_rule, action_molecule)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.critic_optimizer)

            # Delayed policy updates
            if epoch % self.policy_freq == 0:
                with autocast(enabled=self.config.use_amp):
                    # Compute actor loss
                    actor_loss = -self.critic.Q1(state, self.actor(state)[0], self.actor(state)[1]).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                self.scaler.scale(actor_loss).backward()
                self.scaler.step(self.actor_optimizer)

                # Update the frozen target models
                self._update_target_network(self.actor, self.actor_target)
                self._update_target_network(self.critic, self.critic_target)

            self.scaler.update()

            # Update learning rate
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            # wanb.log({"actor_loss": actor_loss, "critic_loss": critic_loss}, step=self.timestep)

    def _update_target_network(self, model, target_model):
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    def select_action(self, state):
        with torch.no_grad():
            rule, mol = self.actor(state)
            return rule, mol

    # Include _update_target_network and other methods as previously defined

