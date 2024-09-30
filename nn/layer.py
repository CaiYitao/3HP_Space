import torch

import torch.nn as nn 
from torch.nn import functional as F
# import torch.nn.functional as F
from torch_geometric.nn import GENConv, GINEConv
from torch_geometric.data import Batch
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from typing import Optional


from torch import Tensor
from torch.nn import Parameter

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax

import math
from torch_geometric.nn import HeteroLinear, GraphNorm
from torch_geometric.utils import softmax, scatter
bond_attr_dim = 3
atom_attr_dim = 10

class GraphGPSLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dim_h = cfg.dim_h
        self.num_heads = cfg.num_heads
        self.attn_dropout = cfg.attn_dropout
        self.layer_norm = cfg.layer_norm
        self.batch_norm = cfg.batch_norm

        # self.edge_emb = nn.Linear(bond_attr_dim, atom_attr_dim)
        # Local message-passing model.
        if cfg.local_gnn_type == 'None':
            self.local_model = None
        elif cfg.local_gnn_type == 'GENConv':
            self.local_model = GENConv(cfg.dim_h, cfg.dim_h)
        elif cfg.local_gnn_type == 'GINE':
            self.local_model = GINEConv(nn.Sequential(nn.Linear(cfg.dim_h, cfg.dim_h), nn.GELU(),nn.Linear(cfg.dim_h,cfg.dim_h)),edge_dim=cfg.bond_attr_dim)

        # Global attention transformer-style model.
        if cfg.global_model_type == 'None':
            self.self_attn = None
        elif cfg.global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(cfg.dim_h, cfg.num_heads, dropout=self.attn_dropout, batch_first=True)

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = nn.LayerNorm(cfg.dim_h)
            self.norm1_attn = nn.LayerNorm(cfg.dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(cfg.dim_h)
            self.norm1_attn = nn.BatchNorm1d(cfg.dim_h)

        self.dropout_local = nn.Dropout(cfg.dropout)
        self.dropout_attn = nn.Dropout(cfg.dropout)

        # Feed Forward block.
        self.activation = F.gelu
        self.ff_linear1 = nn.Linear(cfg.dim_h, cfg.dim_h * 2)
        self.ff_linear2 = nn.Linear(cfg.dim_h * 2, cfg.dim_h)
        if self.layer_norm:
            self.norm2 = nn.LayerNorm(cfg.dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(cfg.dim_h)
        self.ff_dropout1 = nn.Dropout(cfg.dropout)
        self.ff_dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, batch):
        h = batch.x
        # print(f"batch.x shape: {h.shape}")
        # print(f"batch.x: {batch.x}")

        # print(f"batch.edge_index: {batch.edge_index}")
        # print(f"batch.edge_attr: {batch.edge_attr}")
        # print(f"batch.edge_attr shape: {batch.edge_attr.shape} datatype: {batch.edge_attr.dtype}")
        h_in = h.float()  # for first residual connection
        # print(f"h_in shape: {h_in.shape} ")
        h_out_list = []
        # edge_attr = self.edge_emb(batch.edge_attr.float().unsqueeze(0)).squeeze(0)
        # print(f"edge_attr after embeding: {edge_attr}")
        # print(f"edge_attr shape: {edge_attr.shape}")
        # if edge_attr.shape[0] == 0:
        #     edge_attr = edge_attr.view(edge_attr.shape[1:])
        # print(f"edge_attr shape after view: {edge_attr.shape}")
        # Local MPNN with edge attributes.
        if self.local_model is not None:

            # if batch.edge_attr.shape[0] == 0:
            #     h_local = self.local_model(h,batch.edge_index).float()  # No edge (bond) and attributes.
            # else:
            h_local = self.local_model(h_in, batch.edge_index, edge_attr=batch.edge_attr.float()).float()
            # print(f"h_local shape: {h_local.shape} datatype: {h_local.dtype} h_local: {h_local}")
            h_local = self.dropout_local(h_local)

            h_local = h_in + h_local  # Residual connection.
            if self.layer_norm:
                h_local = self.norm1_local(h_local)

            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention or compression.
        if self.self_attn is not None:

            h_attn = self._sa_block(h_in, mask=None)
            # print(f"h_attn shape: {h_attn.shape}")
            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.

        h = sum(h_out_list)


        # Feed Forward block.
        h = h + self._ff_block(h)
        # print(f"h shape after ff_block: {h.shape}")
        if self.layer_norm:
            h = self.norm2(h)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, mask):
        """Self-attention block."""
        x = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)[0]
        return x

    def _ff_block(self, x):
        """Feed Forward block."""
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))



class HyperGraphLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()


        self.dim_h = cfg.dim_h
        self.num_heads = cfg.num_heads
        self.attn_dropout = cfg.attn_dropout
        self.layer_norm = cfg.layer_norm
        self.batch_norm = cfg.batch_norm
        # self.lin = nn.Linear(cfg.dim_h * cfg.hg_attn_heads, cfg.dim_h)
        # self.hypergraph_conv = HypergraphConv(cfg.dim_h, cfg.dim_h, use_attention=True)
        self.hypergraph_conv = DirectedHGConv(cfg)
        self.attn = nn.MultiheadAttention(cfg.dim_h, cfg.num_heads, dropout=self.attn_dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim_h, cfg.dim_h*2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_h*2, cfg.dim_h),
            nn.GELU(),
            nn.Dropout(cfg.dropout)
        )

        self.layer_norm_local = nn.LayerNorm(cfg.dim_h)
        self.layer_norm_attn = nn.LayerNorm(cfg.dim_h)
        self.layer_norm = nn.LayerNorm(cfg.dim_h)
        
    def forward(self, hypergraph):
        h = hypergraph.x
        h_in = h

        # print(f"hypergraph layer h_in shape: {h_in.shape}")
        # print(f"hypergraph edge_attr shape: {hypergraph.edge_attr.shape}")
        # edge_attr = self.hyperedge_emb(hypergraph.edge_attr)
        # print(f"hypergraph layer edge_attr shape after embedding: {edge_attr.shape}")
        # print("hyperedge_index: ", hypergraph.edge_index, "hyperedge_attr shape: ", hypergraph.edge_attr.shape)
        # print("hyperedge_index[1]", hypergraph.edge_index[1] )
        # x_j = torch.index_select(hypergraph.edge_attr, 0, hypergraph.edge_index[1])    
        # print(f"x_j shape: {x_j.shape}") 
        hyperedge_index = transform_hyperedge_index(hypergraph.edge_index)
        # print(f"hyperedge_index after transform: {hyperedge_index}")
        hyperedge_head_tail = hypergraph.edge_index[1]
        h_local = self.hypergraph_conv(h,hyperedge_index,hyperedge_head_tail , hypergraph.edge_attr, hypergraph.batch)
        # h_local = self.hypergraph_conv(h, hypergraph.edge_index, hyperedge_attr = x_j)
        # print(f"hypergraph layer h_local shape after hypergraph conv: {h_local.shape}")
        h_local = h_local + h_in
        h_local = self.layer_norm_local(h_local)
        h_attn = self.attn(h, h, h, need_weights=False)[0]
        # print(f"hypergraph layer h_attn shape after attn: {h_attn.shape}")
        h_attn = h_attn + h_in
        h_attn = self.layer_norm_attn(h_attn)

        h= h_local + h_attn
        h = h + self.mlp(h)
        h = self.layer_norm(h)
        # print(f"hypergraph layer output x shape : {h.shape}")
        hypergraph.x = h
        return hypergraph





# class CrossAttention(nn.Module):
#     def __init__(self, state_dim, rule_dim, output_dim):
#         super(CrossAttention, self).__init__()
#         self.state_proj = nn.Linear(state_dim, output_dim)
#         self.rule_proj = nn.Linear(rule_dim, output_dim)
#         self.score_func = nn.Linear(output_dim, 1)

#     def forward(self, state, rule):
#         """
#         Performs cross attention on state and rule representations.

#         Args:
#             state (torch.Tensor): State representation (batch_size, state_dim).
#             rule (torch.Tensor): Rule representation (batch_size, rule_dim).

#         Returns:
#             torch.Tensor: Weighted sum of rule features (batch_size, output_dim).
#         """
#         # Project state and rule representations to a shared space.
#         state_proj = self.state_proj(state)
#         rule_proj = self.rule_proj(rule)

#         # Calculate attention scores (batch_size, 1).
#         attention_scores = self.score_func(torch.tanh(state_proj + rule_proj.transpose(1, 2)))
#         attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)

#         # Weighted sum of rule features based on attention weights.
#         context_vector = torch.bmm(attention_weights, rule)

#         return context_vector



class CrossAttention(nn.Module):
    def __init__(self, state_dim, rule_dim, output_dim):
        super().__init__()
        self.query = nn.Linear(rule_dim, output_dim)
        self.key = nn.Linear(state_dim, output_dim)
        self.value = nn.Linear(state_dim, output_dim)

    def forward(self, state, rule):
        # Reshape representations if necessary
        # rule = rule.view(rule.size(0), rule.size(1), -1)
        # state = state.view(state.size(0), state.size(1), -1)

        # Compute query, key, and value
        # print(f"rule shape in cross attention: {rule.shape}")
        # print(f"state shape in cross attention: {state.shape}")
        query = self.query(rule).unsqueeze(1)  # Add singleton dimension for batch
        # print(f"query shape in cross attention: {query.shape}")
        key = self.key(state).unsqueeze(0).transpose(-2, -1)   # Transpose key
        # print(f"key shape in cross attention: {key.shape}")
        value = self.value(state).unsqueeze(1)  # Add singleton dimension for batch
        # print(f"value shape in cross attention: {value.shape}")

        # Compute cross-attention scores
        attention_scores = torch.bmm(query, key) / (query.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # print(f"attention_weights shape in cross attention: {attention_weights.shape}")
        # Apply attention to value
        attended_representation = torch.bmm(attention_weights, value)
        # print(f"attended_representation shape in cross attention: {attended_representation.shape}")
        # Combine representations
        combined_representation = rule + attended_representation.squeeze(1)
        # print(f"combined_representation shape in cross attention: {combined_representation.shape}")

        return combined_representation

def transform_hyperedge_index(hyperedge_index):
    if hyperedge_index.shape[1] == 0:  # empty hyperedge_index
        return hyperedge_index
    transition_points = (hyperedge_index[1][:-1] == 1) & (hyperedge_index[1][1:] == 0)
    edge_indices = torch.cumsum(torch.cat([torch.tensor([0]), transition_points]), dim=0)
    return torch.stack([hyperedge_index[0], edge_indices])

class DirectedHGConv(MessagePassing):
    def __init__(self, config):
        super(DirectedHGConv, self).__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.dim_size = config.dim_h
        self.num_edge_heads = config.num_edge_heads
        self.num_node_heads = config.num_node_heads

        self.Q1 = nn.Linear(self.dim_size, self.dim_size, bias=False)
        self.K1 = nn.Linear(self.dim_size, self.dim_size, bias=False)
        self.V1 = nn.Linear(self.dim_size, self.dim_size, bias=False)

        self.edge_linear = nn.Linear(self.dim_size, self.dim_size)

        self.head_tail_linear = HeteroLinear(self.dim_size, self.dim_size, 2)

        self.to_head_tail_linear = HeteroLinear(self.dim_size, self.dim_size, 2)

        self.Q2 = nn.Linear(self.dim_size, self.dim_size, bias=False)
        self.K2 = nn.Linear(self.dim_size, self.dim_size, bias=False)
        self.V2 = nn.Linear(self.dim_size, self.dim_size, bias=False)

        self.u1 = nn.Linear(self.dim_size, self.dim_size)
        self.u2 = nn.Linear(self.dim_size, self.dim_size)

        self.norm = GraphNorm(self.dim_size)

    def forward(self, x, hyperedge_index, edge_in_out_head_tail, edge_attr, batch):
        # x [num_nodes, dim_size] edge_attr [num_edges, dim_size] hyperedge_index [2, num_nodeedges] edge_in_out_head_tail [num_nodeedges]
        # print(f"hyperedge_index shape: {hyperedge_index.shape} edge_attr shape: {edge_attr.shape} ")
        hyperedges = self.edge_updater(hyperedge_index, x=x, edge_attr=edge_attr,
                                       edge_in_out_head_tail=edge_in_out_head_tail)
        # hyperedges [num_edges, dim_size]
        # print(f"hyperedges shape after edge_updater: {hyperedges.shape}")
        edge_attr_out = self.edge_linear(edge_attr)
        # edge_attr_out [num_edges, dim_size]
        # print(f"edge_attr_out shape: {edge_attr_out.shape}")
        hyperedges = hyperedges + edge_attr_out
        # print(f"hyperedges shape after add edge_attr_out: {hyperedges.shape}")
        out = self.propagate(hyperedge_index.flip([0]), x=x, hyperedges=hyperedges,
                             edge_in_out_head_tail=edge_in_out_head_tail, batch=batch)
        return out

    def edge_update(self, edge_index=None, x_j=None, edge_attr_i=None, edge_in_out_head_tail=None):
        
        m = self.head_tail_linear(x_j, edge_in_out_head_tail)
        # print(f"x_j shape: {x_j.shape} ")
        # print(f"x shape after reactant and product weight transformation: {m.shape}")
        # m, edge_attr_i [num_nodeedges, dim_size]
        query = self.Q1(edge_attr_i)
        key = self.K1(m)
        value = self.V1(m)

        query = query.reshape(-1, self.num_edge_heads, self.dim_size // self.num_edge_heads)
        key = key.reshape(-1, self.num_edge_heads, self.dim_size // self.num_edge_heads)
        value = value.reshape(-1, self.num_edge_heads, self.dim_size // self.num_edge_heads)
        # query, key, value [num_nodeedges, num_edge_heads, head_size]
        attn = (query * key).sum(dim=-1)
        attn = attn / math.sqrt(self.dim_size // self.num_edge_heads)
        # print(f"edge update attn shape: {attn.shape} ")
        # attn [num_nodeedges, num_edge_heads]
        attn_score = softmax(attn, edge_index[1])
        # print(f"edge update attn_score  shape: {attn_score.shape} attn_score softmax: {attn_score}")
        attn_score = attn_score.unsqueeze(-1)
        # attn_score [num_nodeedges, num_edge_heads, 1]
        out = value * attn_score
        # out [num_nodeedges, num_edge_heads, head_size]
        out = scatter_add(out, edge_index[1], 0)
        # out [num_edges, num_edge_heads, head_size]
        # print(f"edge update out shape: {out.shape} ")
        out = out.reshape(-1, self.dim_size)

        return out

    def message(self, edge_index=None, x_i=None, hyperedges_j=None, edge_in_out_head_tail=None):
        # print(f"hyperedges_j shape: {hyperedges_j.shape}")
        m = self.to_head_tail_linear(hyperedges_j, edge_in_out_head_tail)
        # print(f"hyperedge_j after head tail linear transofrm shape: {m.shape}")
        # m, x_i [num_nodeedges, dim_size]
        query = self.Q2(x_i)
        key = self.K2(m)
        value = self.V2(m)

        query = query.reshape(-1, self.num_node_heads, self.dim_size // self.num_node_heads)
        key = key.reshape(-1, self.num_node_heads, self.dim_size // self.num_node_heads)
        value = value.reshape(-1, self.num_node_heads, self.dim_size // self.num_node_heads)
        # query, key, value [num_nodeedges, num_node_heads, head_size]
        attn = (query * key).sum(dim=-1)
        # attn [num_nodeedges, num_node_heads]
        attn = attn / math.sqrt(self.dim_size // self.num_node_heads)
        # print(f"message attn shape: {attn.shape} message attn: {attn}")
        attn_score = softmax(attn, edge_index[0])

        attn_score = attn_score.unsqueeze(-1)
        print(f"message attn_score shape after unsqueeze: {attn_score.shape} message attn_score: {attn_score}")
        # attn_score [num_nodeedges, num_node_heads, 1]
        out = value * attn_score
        # print(f"message out shape: {out.shape} ")
        # out [num_nodeedges, num_node_heads, head_size]

        return out

    def update(self, inputs, x=None, batch=None):
        # print(f"inputs shape: {inputs.shape}")
        inputs = inputs.reshape(-1, self.dim_size)
        # x, inputs [num_nodes, dim_size]
        inputs = self.u2(inputs)
        x = self.u1(x)
        out = inputs + x
        out = self.norm(out, batch)
        out = F.elu(out)
        return out
    



class HypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W} \in \mathbb{R}^M` is the diagonal hyperedge
    weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.

    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:

    .. code-block:: python

        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        attention_mode (str, optional): The mode on how to compute attention.
            If set to :obj:`"node"`, will compute attention scores of nodes
            within all nodes belonging to the same hyperedge.
            If set to :obj:`"edge"`, will compute attention scores of nodes
            across all edges holding this node belongs to.
            (default: :obj:`"node"`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          hyperedge indices :math:`(|\mathcal{V}|, |\mathcal{E}|)`,
          hyperedge weights :math:`(|\mathcal{E}|)` *(optional)*
          hyperedge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_mode: str = 'node',
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        assert attention_mode in ['node', 'edge']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode = attention_mode

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    @disable_dynamic_shapes(required_args=['num_edges'])
    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None,
                num_edges: Optional[int] = None) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (torch.Tensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (torch.Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (torch.Tensor, optional): Hyperedge feature matrix
                in :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
            num_edges (int, optional) : The number of edges :math:`M`.
                (default: :obj:`None`)
        """
        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            print(f"x_i shape: {x_i.shape} x_i: {x_i}")
            print(f"hyperedge_attr shape in HypergraphConv: {hyperedge_attr.shape} hyperedge_attr in HypergraphConv: {hyperedge_attr}")
            print(f"hyperedge_index: {hyperedge_index} hyperedge_index[1]: {hyperedge_index[1]}")
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == 'node':
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out