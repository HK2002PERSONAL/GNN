import torch
import torch.nn as nn

from model_blocks import (
    GCN_InfoShare_block,
)

# pylint:disable=W0223
class FClipBase(nn.Module):
    """带有 clip 功能的基类"""
    def __init__(self, clip=5):
        super().__init__()
        self.clip = clip

    def fclip(self, x: torch.Tensor):
        if self.clip < 1:
            z = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-5)
            lo = torch.quantile(z, self.clip, dim=0, keepdim=True)
            hi = torch.quantile(z, 1.0 - self.clip, dim=0, keepdim=True)
            return torch.clip(z, lo, hi)
        else:
            return torch.clip(
                (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-5), 
                min=-self.clip, max=self.clip)



class HIST_3GCN(FClipBase):
    """
    model HyperGCN Model 
    GCN + GCN + GCN
    """
    def __init__(
        self, 
        d_feat=6, hidden_size=64, clip=5,
        ew1: bool = True, ew2: bool = True, ew3: bool = True
    ):
        super().__init__(clip=clip)

        self.d_feat = d_feat
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(d_feat, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        self.predefined_info_block_1 = GCN_InfoShare_block(hidden_size, ew=ew1)
        self.info_gate_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.info_gate_1.weight)
        self.predefined_info_block_2 = GCN_InfoShare_block(hidden_size, ew=ew2)
        self.info_gate_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.info_gate_2.weight)
        self.predefined_info_block_3 = GCN_InfoShare_block(hidden_size, ew=ew3)
        self.info_gate_3 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.info_gate_3.weight)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        

    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        concept_matrix_3: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        # Layer 1
        p1_info = self.predefined_info_block_1(x_hidden, concept_matrix_1, market_value)
        x_hidden_dp1 = x_hidden - p1_info

        # Layer 2
        p2_info = self.predefined_info_block_2(x_hidden_dp1, concept_matrix_2, market_value)
        x_hidden_dp1p2 = x_hidden_dp1 - p2_info

        # Layer 3
        p3_info = self.predefined_info_block_3(x_hidden_dp1p2, concept_matrix_3, market_value)
        x_hidden_dp1p2p3 = x_hidden_dp1p2 - p3_info

        # final layer
        i_info = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2p3))

        # output layer
        p1_info = self.info_gate_1(p1_info)
        p2_info = self.info_gate_2(p2_info)
        p3_info = self.info_gate_3(p3_info)
        
        all_info = p1_info + p2_info + p3_info + i_info

        return self.fc_out(all_info)


class HIST_Orth(HIST_3GCN):
    """
    model Orth Model 
    GCN + GCN + GCN
    """

    def orth(
        self,
        input_info: torch.Tensor, gnn_info: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:

        x, g = input_info, gnn_info

        x = x - x.mean(dim=0, keepdim=True)
        g = g - g.mean(dim=0, keepdim=True)

        beta = (x * g).sum(dim=0, keepdim=True) / (g.square().sum(dim=0, keepdim=True) + eps)

        return input_info - beta * gnn_info
        

    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        concept_matrix_3: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        # Layer 1
        p1_info = self.predefined_info_block_1(x_hidden, concept_matrix_1, market_value)
        x_hidden_dp1 = self.orth(x_hidden, p1_info)

        # Layer 2
        p2_info = self.predefined_info_block_2(x_hidden_dp1, concept_matrix_2, market_value)
        x_hidden_dp1p2 = self.orth(x_hidden_dp1, p2_info)

        # Layer 3
        p3_info = self.predefined_info_block_3(x_hidden_dp1p2, concept_matrix_3, market_value)
        x_hidden_dp1p2p3 = self.orth(x_hidden_dp1p2, p3_info)

        # final layer
        i_info = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2p3))

        # output layer
        p1_info = self.info_gate_1(p1_info)
        p2_info = self.info_gate_2(p2_info)
        p3_info = self.info_gate_3(p3_info)

        all_info = p1_info + p2_info + p3_info + i_info

        return self.fc_out(all_info)
