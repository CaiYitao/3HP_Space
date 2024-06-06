import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model.layer import GraphGPSLayer, HyperGraphLayer,CrossAttention
from featurizer import HyperGraphFeaturizer, RuleGraphFeaturizer,MolGraphFeaturizer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GENConv, GINEConv,HypergraphConv
from torch_geometric.nn.pool import SAGPooling, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data.hypergraph_data import HyperGraphData
from torch.distributions import Categorical
from torch.cuda.amp import GradScaler, autocast
import copy
from collections import deque
import random
from buffer import ReplayBuffer,choose_optimizer,choose_lr_scheduler
import wandb
from encoder import GraphEncoder

class HyperGraphModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.hypergraph_featurizer = HyperGraphFeaturizer(cfg)
        self.hypergraph_layer = HyperGraphLayer(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim_h, cfg.dim_h*2),
                                nn.GELU(),
                                nn.Dropout(cfg.dropout),
                                nn.Linear(cfg.dim_h*2, cfg.dim_h))

    def forward(self, dg):
        hypergraph_data = self.hypergraph_featurizer(dg)
       
        outputs = []
        # x = hypergraph_data.x
        # print(f"hypergraph data x shape: {x.shape}")
        # print(f"hypergraph data original batch: {hypergraph_data.batch}")
        for i in range(self.config.num_hglayers):
            hypergraph_data = self.hypergraph_layer(hypergraph_data)
            x = hypergraph_data.x.squeeze(0)
            outputs.append(x)
        # print(f"x shape: {x.shape}")
        if self.config.parallel_output:
            output = torch.sum(torch.stack(outputs), 0)
  
        else:
            output = x
        hypergraph_data.batch = torch.zeros(hypergraph_data.x.shape[0], dtype = torch.long)
        # print(f"hypergraph data manually created batch shape: {hypergraph_data.batch}")

        output = global_mean_pool(output, hypergraph_data.batch)
        # output = self.mlp(output)
        # print(f"hypergraph output shape: {output.shape}")
        return output





class RuleActor(nn.Module):
    def __init__(self, config):
        super(RuleActor, self).__init__()
        self.config = config
        self.hypergraph_model = HyperGraphModel(config)
        self.mlp = nn.Sequential(nn.Linear(config.dim_h, config.dim_h*2),
                                        nn.GELU(),
                                        nn.Dropout(config.dropout),
                                        nn.Linear(config.dim_h*2, config.dim_h),
                                        nn.GELU(),
                                        nn.Dropout(config.dropout),
                                        nn.Linear(config.dim_h, config.rule_dim))


    def forward(self,dg_state):
        rep = self.hypergraph_model(dg_state)
        # print(f"rep shape: {rep.shape}")
        r_out = self.mlp(rep)
        # print(f"output shape: {r_out.shape}")
        return torch.softmax(r_out, dim = -1)


class MolActor(nn.Module):
    rule_idx_dict = {0: "R1: (S)-lactate + NAD+ = pyruvate + NADH + H+",
                    1:"R2: Ac-aldehyde + CoA + NAD+ = Ac-CoA + NADH + H+",
                    2:"R3: L-Aspartate + 2-Oxoglutarate <=> Oxaloacetate + L-Glutamate",
                    3:"R4: Acyl-CoA + H2O <=> CoA + Carboxylate",
                    4:"R5: SAM + CO2 = SAM-CO2H",
                    5:"R6: L-malate = fumarate + H2O",
                    6:"R7: Lactoyl-CoA <=> Propenoyl-CoA + H2O",
                    7:"R8: L-Aspartate <=> Fumarate + Ammonia",
                    8:"R9: L-lysine = (3S)-3,6-diaminohexanoate",
                    9:"R10: ATP + Acetate + CoA <=> AMP + PPi + Ac-CoA",
                    10:"R11: ATP + Ac-CoA + HCO3- = ADP + malonyl-CoA + Pi"}
    
    def __init__(self, config):
        super(MolActor, self).__init__()
        self.config = config
        self.hypergraph_model = HyperGraphModel(config)
        self.rule_encoder = GraphEncoder(config,rule_graph=True)
        self.rule_featurizer = RuleGraphFeaturizer()
        self.cross_attention = CrossAttention(config.dim_h, config.dim_h, config.dim_h)
        self.mlp = nn.Sequential(nn.Linear(config.dim_h, config.dim_h*2),
                                        nn.GELU(),
                                        nn.Dropout(config.dropout),
                                        nn.Linear(config.dim_h*2, config.dim_h),
                                        nn.GELU(),
                                        nn.Dropout(config.dropout),
                                        nn.Linear(config.dim_h, config.mol_dim))


    def forward(self,dg_state,rule):
        # print(f"rule: {rule}")
        rule_name = self.rule_idx_dict[rule]
        # print(f"rule_name: {rule_name}")
        rule_rep = self.rule_encoder(self.rule_featurizer(rule_name))
        # print(f"rule_rep shape: {rule_rep.shape}")
        state_rep = self.hypergraph_model(dg_state)
        # print(f"state_rep shape: {state_rep.shape}")
        attended_output = self.cross_attention(state_rep, rule_rep)
        # print(f"attended_output shape: {attended_output.shape}")
        output = self.mlp(attended_output)
        return torch.softmax(output, dim = -1)  
