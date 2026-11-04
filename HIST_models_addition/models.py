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



class HIST_FactorGCL(FClipBase):
    """
    model FactorGCL 
    GCN + GCN + GCN
    dual output for contractive learning

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
        self.predefined_info_block_2 = GCN_InfoShare_block(hidden_size, ew=ew2)
        self.predefined_info_block_3 = GCN_InfoShare_block(hidden_size, ew=ew3)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)


    def real_forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        concept_matrix_3: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        # Layer 1
        p1_info_fore, p1_info_back = self.predefined_info_block_1(x_hidden, concept_matrix_1, market_value)
        x_hidden_dp1 = x_hidden - p1_info_back

        # Layer 2
        p2_info_fore, p2_info_back = self.predefined_info_block_2(x_hidden_dp1, concept_matrix_2, market_value)
        x_hidden_dp1p2 = x_hidden_dp1 - p2_info_back

        # Layer 3
        p3_info_fore, p3_info_back = self.predefined_info_block_3(x_hidden_dp1p2, concept_matrix_3, market_value)
        x_hidden_dp1p2p3 = x_hidden_dp1p2 - p3_info_back

        # final layer
        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2p3))

        # output layer
        all_info_fore = p1_info_fore + p2_info_fore + p3_info_fore + i_info_fore

        return self.fc_out(all_info_fore), i_info_fore

    
    def goof_forward(
        self,
        x_fut: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        concept_matrix_3: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
    

        with torch.no_grad():
            # encode
            x_hidden = self.encoder(self.fclip(x_fut))
            # Layer 1
            _, p1_info_back = self.predefined_info_block_1(x_hidden, concept_matrix_1, market_value)
            x_hidden_dp1 = x_hidden - p1_info_back

            # Layer 2
            _, p2_info_back = self.predefined_info_block_2(x_hidden_dp1, concept_matrix_2, market_value)
            x_hidden_dp1p2 = x_hidden_dp1 - p2_info_back

            # Layer 3
            _, p3_info_back = self.predefined_info_block_3(x_hidden_dp1p2, concept_matrix_3, market_value)
            x_hidden_dp1p2p3 = x_hidden_dp1p2 - p3_info_back

            # final layer
            i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2p3))

        return i_info_fore


    def forward(
        self,
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        concept_matrix_3: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
        x: torch.Tensor, # [N, F]
        x_fut: torch.Tensor | None = None, # [N, F]
        output: bool = False
    ):
        if output:
            batch_result, _ = self.real_forward(
                x, concept_matrix_1, concept_matrix_2, concept_matrix_3, market_value)
            return batch_result

        else:
            batch_result, real_indi_info = self.real_forward(
                x, concept_matrix_1, concept_matrix_2, concept_matrix_3, market_value)
            goof_indi_info = self.goof_forward(
                x_fut, concept_matrix_1, concept_matrix_2, concept_matrix_3, market_value)

            return batch_result, real_indi_info, goof_indi_info



class HIST_FactorGCL_T(HIST_FactorGCL):

    def real_forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        concept_matrix_3: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        # Layer 1
        p1_info_fore, p1_info_back = self.predefined_info_block_1(x_hidden, concept_matrix_1, market_value)
        x_hidden_dp1 = x_hidden - p1_info_back

        # Layer 2
        p2_info_fore, p2_info_back = self.predefined_info_block_2(x_hidden_dp1, concept_matrix_2, market_value)
        x_hidden_dp1p2 = x_hidden_dp1 - p2_info_back

        # Layer 3
        p3_info_fore, p3_info_back = self.predefined_info_block_3(x_hidden_dp1p2, concept_matrix_3, market_value)
        x_hidden_dp1p2p3 = x_hidden_dp1p2 - p3_info_back

        # final layer
        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2p3))

        # output layer
        all_info_fore = p1_info_fore + p2_info_fore + p3_info_fore + i_info_fore

        return self.fc_out(all_info_fore), all_info_fore

    
    def goof_forward(
        self,
        x_fut: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        concept_matrix_3: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
    
        with torch.no_grad():
            return self.real_forward(
                x_fut, 
                concept_matrix_1, 
                concept_matrix_2, 
                concept_matrix_3, 
                market_value
            )


    def forward(
        self,
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        concept_matrix_3: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
        x: torch.Tensor, # [N, F]
        x_fut: torch.Tensor | None = None, # [N, F]
        output: bool = False
    ):
        if output:
            batch_result, _ = self.real_forward(
                x, concept_matrix_1, concept_matrix_2, concept_matrix_3, market_value)
            return batch_result

        else:
            batch_result, real_all_info = self.real_forward(
                x, concept_matrix_1, concept_matrix_2, concept_matrix_3, market_value)
            _, goof_all_info = self.goof_forward(
                x_fut, concept_matrix_1, concept_matrix_2, concept_matrix_3, market_value)

            return batch_result, real_all_info, goof_all_info


class HIST_FactorGCL_T_Factor(HIST_FactorGCL_T):

    def forward(
        self,
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        concept_matrix_3: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
        x: torch.Tensor, # [N, F]
        x_fut: torch.Tensor | None = None, # [N, F]
        output: bool = False
    ):
        if output:
            real_output, _ = self.real_forward(
                x, concept_matrix_1, concept_matrix_2, concept_matrix_3, market_value)
            return real_output

        else:
            real_output, _ = self.real_forward(
                x, concept_matrix_1, concept_matrix_2, concept_matrix_3, market_value)
            goof_output, _ = self.goof_forward(
                x_fut, concept_matrix_1, concept_matrix_2, concept_matrix_3, market_value)

            return real_output, goof_output