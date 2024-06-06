
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layer import GraphGPSLayer, HyperGraphLayer,CrossAttention

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

import wandb




class GraphEncoder(nn.Module):
    def __init__(self, cfg,rule_graph=False):
        super(GraphEncoder, self).__init__()
        self.config = cfg
        if rule_graph == True:
            self.linear = nn.Linear(cfg.atom_attr_dim + 4, cfg.dim_h)
        else:
            self.linear = nn.Linear(cfg.atom_attr_dim, cfg.dim_h)

        self.layer = GraphGPSLayer(cfg)
        self.pool = SAGPooling(cfg.dim_h, ratio=0.5)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim_h, cfg.dim_h*2),
                                nn.GELU(),
                                nn.Dropout(cfg.dropout),
                                nn.Linear(cfg.dim_h*2, cfg.dim_h))
    
    def forward(self, batch):
        outputs = []

        # print(f"mol graph batch: {batch}")
        # print(f"self.linear: {self.linear}")
        # print(f"mol graph x shape: {batch.x.shape}")
        batch.x = self.linear(batch.x.float())
        # print(f"mol graph x shape: {batch.x.shape}")
        # print(f"mol graph batch shape: {batch.batch}")
        for i in range(self.config.num_layers):
            batch = self.layer(batch)
            outputs.append(batch.x.squeeze(0))
        if self.config.parallel_output:
            output = torch.sum(torch.stack(outputs), 0)
            
        else: 
            output = batch.x
        
        # print(f"mol graph output shape: {output.shape}")
        # batch.batch = torch.zeros(batch.x.shape[0], dtype = torch.long)
        # print(f"mol graph batch shape after manual batch: {batch.batch}")
        output  =  global_mean_pool(output, batch.batch)
        # print(f"output shape after global mean pool: {output.shape}")
        output = self.mlp(output)
        return output
        