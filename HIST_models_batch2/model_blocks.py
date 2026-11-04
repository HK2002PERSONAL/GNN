from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, LayerNorm

class HIST_block_base(nn.Module):

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size

        self.fc_info = nn.Linear(hidden_size, hidden_size)  
        torch.nn.init.xavier_uniform_(self.fc_info.weight)
        self.leaky_relu = nn.LeakyReLU()
        self.fc_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_fore.weight)
        self.fc_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_fore.weight)
        

    def cal_cos_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
        """the 2nd dimension of x and y are the same"""
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
        cos_similarity = xy/x_norm.mm(torch.t(y_norm))
        cos_similarity[cos_similarity != cos_similarity] = 0
        return cos_similarity

    
    def forward(self):
        raise NotImplementedError(
            "base class of HIST block shouldn't be used directly")


class GCN_InfoShare_block(HIST_block_base):
    """市值加权边HyperGCN结构"""

    def __init__(self, hidden_size: int = 64, dropout: float = 0.0, ew: bool = True):
        super().__init__(hidden_size=hidden_size)
        self.conv1 = GCNConv(hidden_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.norm1 = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.ew = ew

    @staticmethod
    def _build_bipartite_graph(
        concept_matrix: torch.Tensor,  # [N, C]
        market_value: torch.Tensor,    # [N]
    ):
        N = concept_matrix.shape[0]

        valid = concept_matrix.sum(0) != 0
        cm = concept_matrix[:, valid]
        n_concepts = int(cm.shape[1])

        stock_idx, concept_idx = torch.nonzero(cm, as_tuple=True)
        concept_idx = concept_idx + N

        # 第一阶段：股票 -> 概念
        edge_s2c = torch.stack([stock_idx, concept_idx], dim=0)

        # 第二阶段：概念 -> 股票
        edge_c2s = edge_s2c.flip(0)

        mv = market_value.detach().clamp_min(0.0)
        mv = mv / (mv.sum() + 1e-8) * N
        edge_weight = mv[stock_idx]

        return edge_s2c, edge_c2s, edge_weight, n_concepts

    # pylint:disable=W0221
    def forward(
        self,
        x_hidden: torch.Tensor,     # [N, H]
        concept_matrix: torch.Tensor,  # [N, C]
        market_value: torch.Tensor,    # [N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = x_hidden.shape[0]

        edge_s2c, edge_c2s, edge_weight, n_concepts = (
            self._build_bipartite_graph(concept_matrix, market_value)
        )

        if n_concepts == 0 or edge_s2c.numel() == 0:
            raise ValueError("No Concept Available")

        x_all = torch.cat([x_hidden, x_hidden.new_zeros(n_concepts, self.hidden_size)], dim=0)

        # 第一阶段：只进行 stock -> concept
        h = self.conv1(x_all, edge_s2c, edge_weight) if self.ew else self.conv1(x_all, edge_s2c)
        h = self.norm1(h + x_all)
        h = self.drop(self.leaky_relu(h))

        # 第二阶段：只进行 concept -> stock
        h = self.conv2(h, edge_c2s, edge_weight) if self.ew else self.conv2(h, edge_c2s)

        x_shared = self.norm2(h[:N] + x_hidden)

        # 解码信息
        layer_info = self.fc_info(x_shared)
        layer_info = self.leaky_relu(layer_info)
        layer_info = self.fc_fore(layer_info)

        return layer_info



class Risk_Attention_block(HIST_block_base):

    def __init__(
        self, 
        risk_input: int = 11,
        hidden_size: int = 64,
        head: int = 2,
    ):
        super().__init__(hidden_size=hidden_size)
        self.query = nn.Linear(risk_input, hidden_size)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=head)

    # pylint:disable=W0221
    def forward(
        self,
        x_hidden: torch.Tensor, # [N, H]
        risk_feature: torch.Tensor, # [N, R]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        risk_hidden = self.query(risk_feature)
        x_shared, _ = self.mha(risk_hidden, risk_hidden, x_hidden)

        # 解码信息
        layer_info = self.fc_info(x_shared)
        layer_info = self.leaky_relu(layer_info)
        layer_info = self.fc_fore(layer_info)

        return layer_info

