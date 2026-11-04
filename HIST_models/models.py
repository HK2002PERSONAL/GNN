import torch
import torch.nn as nn

from model_blocks import (
    HIST_PreDefinedConcept_block,
    HIST_HiddenConcept_block,
    GAT_InfoShare_block,
    GCN_InfoShare_block,
    HLGCN_InfoShare_block,
    HLGCN_InfoShare_block_v2,
    Risk_Attention_block,
    RGAT_InfoShare_block
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


class HIST_a0(FClipBase):
    """model 0 原始 HIST 结构 (Predefined Concept + Hidden Concept)"""
    def __init__(self, d_feat=6, hidden_size=64, K=3, clip=5):
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

        self.predefined_info_block = HIST_PreDefinedConcept_block(hidden_size)
        self.hidden_info_block = HIST_HiddenConcept_block(hidden_size, K=K)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)


    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        p_info_fore, p_info_back = self.predefined_info_block(
            x_hidden, concept_matrix, market_value)

        x_hidden_dp = x_hidden - p_info_back

        h_info_fore, h_info_back = self.hidden_info_block(x_hidden_dp)

        x_hidden_dph = x_hidden_dp - h_info_back

        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dph))

        all_info_fore = p_info_fore + h_info_fore + i_info_fore

        return self.fc_out(all_info_fore)



class HIST_a1(FClipBase):
    """model 1 仅保留Predefined Concept"""
    def __init__(self, d_feat=6, hidden_size=64, clip=5):
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

        self.predefined_info_block = HIST_PreDefinedConcept_block(hidden_size)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)


    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        p_info_fore, p_info_back = self.predefined_info_block(
            x_hidden, concept_matrix, market_value)

        x_hidden_i = x_hidden - p_info_back

        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_i))

        all_info_fore = p_info_fore + i_info_fore

        return self.fc_out(all_info_fore)



class HIST_a2(FClipBase):
    """model 2 仅保留Hidden Concept"""
    def __init__(self, d_feat=6, hidden_size=64, clip=5):
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

        self.hidden_info_block = HIST_HiddenConcept_block(hidden_size)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_hidden = self.encoder(self.fclip(x))

        h_info_fore, h_info_back = self.hidden_info_block(x_hidden)

        x_hidden_i = x_hidden - h_info_back

        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_i))

        all_info_fore = h_info_fore + i_info_fore

        return self.fc_out(all_info_fore)



class HIST_a3(FClipBase):
    """model 3 Hidden Concept 替换为GAT"""
    def __init__(self, d_feat=6, hidden_size=64, clip=5):
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

        self.predefined_info_block = HIST_PreDefinedConcept_block(hidden_size)
        self.hidden_info_block = GAT_InfoShare_block(hidden_size)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)

    
    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        p_info_fore, p_info_back = self.predefined_info_block(
            x_hidden, concept_matrix, market_value)

        x_hidden_dp = x_hidden - p_info_back

        h_info_fore, h_info_back = self.hidden_info_block(x_hidden_dp)

        x_hidden_dph = x_hidden_dp - h_info_back

        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dph))

        all_info_fore = p_info_fore + h_info_fore + i_info_fore

        return self.fc_out(all_info_fore)




class HIST_a4(FClipBase):
    """model 4 Predefined Concept 替换为GCN"""
    def __init__(self, d_feat=6, hidden_size=64, clip=5):
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

        self.predefined_info_block = GCN_InfoShare_block(hidden_size)
        self.hidden_info_block = HIST_HiddenConcept_block(hidden_size)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)


    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        p_info_fore, p_info_back = self.predefined_info_block(
            x_hidden, concept_matrix, market_value)

        x_hidden_dp = x_hidden - p_info_back

        h_info_fore, h_info_back = self.hidden_info_block(x_hidden_dp)

        x_hidden_dph = x_hidden_dp - h_info_back

        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dph))

        all_info_fore = p_info_fore + h_info_fore + i_info_fore

        return self.fc_out(all_info_fore)



class HIST_a5(FClipBase):
    """model 5 GCN + GAT"""
    def __init__(self, d_feat=6, hidden_size=64, clip=5):
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

        self.predefined_info_block = GCN_InfoShare_block(hidden_size)
        self.hidden_info_block = GAT_InfoShare_block(hidden_size)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)


    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        p_info_fore, p_info_back = self.predefined_info_block(
            x_hidden, concept_matrix, market_value)

        x_hidden_dp = x_hidden - p_info_back

        h_info_fore, h_info_back = self.hidden_info_block(x_hidden_dp)

        x_hidden_dph = x_hidden_dp - h_info_back

        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dph))

        all_info_fore = p_info_fore + h_info_fore + i_info_fore

        return self.fc_out(all_info_fore)



class HIST_a6(FClipBase):
    """model 6 GCN + GCN

    参数:
        ew1: 第一层 GCN 是否使用 edge_weight
        ew2: 第二层 GCN 是否使用 edge_weight
    """
    def __init__(self, d_feat=6, hidden_size=64, clip=5, ew1=True, ew2=True):
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

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)


    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        p1_info_fore, p1_info_back = self.predefined_info_block_1(
            x_hidden, concept_matrix_1, market_value)

        x_hidden_dp1 = x_hidden - p1_info_back

        p2_info_fore, p2_info_back = self.predefined_info_block_2(
            x_hidden_dp1, concept_matrix_2, market_value)

        x_hidden_dp1p2 = x_hidden_dp1 - p2_info_back

        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2))

        all_info_fore = p1_info_fore + p2_info_fore + i_info_fore

        return self.fc_out(all_info_fore)


class HIST_a6_v2(HIST_a6):
    """model 6 GCN + HLGCN

    参数:
        ew1: 第一层 GCN 是否使用 edge_weight
        ew2: 第二层 HLGCN 是否使用 edge_weight
    """
    # pylint:disable=W0231
    def __init__(self, d_feat=6, hidden_size=64, clip=5, ew1=True, ew2=True):
        # pylint:disable=W0233
        FClipBase.__init__(self, clip=clip)

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
        self.predefined_info_block_2 = HLGCN_InfoShare_block(hidden_size, ew=ew2)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)


class HIST_a6_v2_2(HIST_a6):
    """model 6 GCN + HLGCN_V2

    参数:
        ew1: 第一层 GCN 是否使用 edge_weight
        ew2: 第二层 HLGCN_V2 是否使用 edge_weight
    """
    def __init__(self, d_feat=6, hidden_size=64, clip=5, ew1=True, ew2=True):
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
        self.predefined_info_block_2 = HLGCN_InfoShare_block_v2(hidden_size, ew=ew2)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)



class HIST_a7(FClipBase):
    """model 7 GCN + RAT（风险注意力）

    Predefined Concept (v0 行业, GCN) + Risk Attention (RAT)。
    风险因子在模型内部做截面 z-score 并 clip 到 [-5, 5]（build 时保留原始值）。
    """
    def __init__(self, d_feat=6, hidden_size=64, clip=5, risk_input=11):
        super().__init__(clip=clip)

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.risk_input = risk_input

        self.encoder = nn.Sequential(
            nn.Linear(d_feat, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        self.predefined_info_block = GCN_InfoShare_block(hidden_size)
        self.risk_info_block = Risk_Attention_block(
            risk_input=risk_input, hidden_size=hidden_size)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)

    def zscore_risk(self, risk_feature: torch.Tensor) -> torch.Tensor:
        """截面按列 z-score 后 clip 到 [-5, 5]"""
        z = (risk_feature - risk_feature.mean(dim=0, keepdim=True)) / (
            risk_feature.std(dim=0, keepdim=True) + 1e-5)
        return z.clamp(-5, 5)

    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix: torch.Tensor, # [N, C]
        market_value: torch.Tensor, # [N]
        risk_feature: torch.Tensor, # [N, R]
    ) -> torch.Tensor:

        x_hidden = self.encoder(self.fclip(x))

        p_info_fore, p_info_back = self.predefined_info_block(
            x_hidden, concept_matrix, market_value)

        x_hidden_dp = x_hidden - p_info_back

        risk_z = self.zscore_risk(risk_feature)
        r_info_fore, r_info_back = self.risk_info_block(x_hidden_dp, risk_z)

        x_hidden_dpr = x_hidden_dp - r_info_back

        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dpr))

        all_info_fore = p_info_fore + r_info_fore + i_info_fore

        return self.fc_out(all_info_fore)



class HIST_a8(FClipBase):
    """model 8 GCN + RGAT（风险图注意力）

    Predefined Concept (v0 行业, GCN) + Risk GAT (RGAT)。
    RGAT 用风险因子暴露的余弦相似度构建 TopK 图，在隐藏表征上做 GAT 消息传递。
    风险因子在模型内部做截面 z-score 并 clip 到 [-5, 5]（build 时保留原始值）。
    """
    def __init__(self, d_feat=6, hidden_size=64, clip=5):
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

        self.predefined_info_block = GCN_InfoShare_block(hidden_size)
        self.risk_info_block = RGAT_InfoShare_block(hidden_size)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)

    def zscore_risk(self, risk_feature: torch.Tensor) -> torch.Tensor:
        """截面按列 z-score 后 clip 到 [-5, 5]"""
        z = (risk_feature - risk_feature.mean(dim=0, keepdim=True)) / (
            risk_feature.std(dim=0, keepdim=True) + 1e-5)
        return z.clamp(-5, 5)

    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix: torch.Tensor, # [N, C]
        market_value: torch.Tensor, # [N]
        risk_feature: torch.Tensor, # [N, R]
    ) -> torch.Tensor:

        x_hidden = self.encoder(self.fclip(x))

        p_info_fore, p_info_back = self.predefined_info_block(
            x_hidden, concept_matrix, market_value)

        x_hidden_dp = x_hidden - p_info_back

        risk_z = self.zscore_risk(risk_feature)
        r_info_fore, r_info_back = self.risk_info_block(x_hidden_dp, risk_z)

        x_hidden_dpr = x_hidden_dp - r_info_back

        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dpr))

        all_info_fore = p_info_fore + r_info_fore + i_info_fore

        return self.fc_out(all_info_fore)



class HIST_a9(FClipBase):
    """
    GCN GCN DRAT
    两阶段训练，用于剥离风格
    """
    def __init__(self, d_feat=6, hidden_size=64, clip=5, risk_input=11):
        super().__init__(clip=clip)

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.risk_input = risk_input

        self.encoder = nn.Sequential(
            nn.Linear(d_feat, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        self.predefined_info_block_1 = GCN_InfoShare_block(hidden_size)
        self.predefined_info_block_2 = GCN_InfoShare_block(hidden_size)
        self.risk_info_block = Risk_Attention_block(
            risk_input=risk_input, hidden_size=hidden_size)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)

    def zscore_risk(self, risk_feature: torch.Tensor) -> torch.Tensor:
        """截面按列 z-score 后 clip 到 [-5, 5]"""
        z = (risk_feature - risk_feature.mean(dim=0, keepdim=True)) / (
            risk_feature.std(dim=0, keepdim=True) + 1e-5)
        return z.clamp(-5, 5)

    def fresh_fc_out(self) -> None:
        torch.nn.init.xavier_uniform_(self.fc_out.weight)
    
    def hiar_info(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N]
        risk_feature: torch.Tensor, # [N, R]
    ):

        x_hidden = self.encoder(self.fclip(x))

        p1_info_fore, p1_info_back = self.predefined_info_block_1(
            x_hidden, concept_matrix_1, market_value)

        x_hidden_dp1 = x_hidden - p1_info_back

        p2_info_fore, p2_info_back = self.predefined_info_block_2(
            x_hidden_dp1, concept_matrix_2, market_value)

        x_hidden_dp1p2 = x_hidden_dp1 - p2_info_back

        risk_z = self.zscore_risk(risk_feature)
        r_info_fore, r_info_back = self.risk_info_block(x_hidden_dp1p2, risk_z)

        x_hidden_dp1p2r = x_hidden_dp1p2 - r_info_back

        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2r))

        return p1_info_fore, p2_info_fore, r_info_fore, i_info_fore

    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N]
        risk_feature: torch.Tensor, # [N, R]
        warm_up: bool = True
    ) -> torch.Tensor:

        if warm_up:
            p1_info_fore, p2_info_fore, r_info_fore, i_info_fore = self.hiar_info(
                x, concept_matrix_1, concept_matrix_2, market_value, risk_feature)
            all_info_fore = p1_info_fore + p2_info_fore + r_info_fore + i_info_fore
            return self.fc_out(all_info_fore)

        else:
            with torch.no_grad():
                p1_info_fore, p2_info_fore, _, i_info_fore = self.hiar_info(
                    x, concept_matrix_1, concept_matrix_2, market_value, risk_feature)

            none_risk_info = p1_info_fore + p2_info_fore + i_info_fore
            return self.fc_out(none_risk_info)


class HIST_a9_v2(HIST_a9):

    def __init__(self, d_feat=6, hidden_size=64, clip=5, risk_input=11):
        super().__init__(d_feat, hidden_size, clip, risk_input)

        self.risk_fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.risk_fc_out.weight)

    # pylint:disable=W0221
    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N]
        risk_feature: torch.Tensor, # [N, R]
    ) -> torch.Tensor:

        p1_info_fore, p2_info_fore, r_info_fore, i_info_fore = self.hiar_info(
            x, concept_matrix_1, concept_matrix_2, market_value, risk_feature)

        none_risk_output = self.fc_out(p1_info_fore + p2_info_fore + i_info_fore)
        risk_only_output = self.risk_fc_out(r_info_fore)

        return none_risk_output, risk_only_output


class HIST_a10(FClipBase):
    """model 10 GCN[Industry]GCN[Risk] + GCN[corr]"""
    def __init__(self, d_feat=6, hidden_size=64, clip=5):
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

        self.predefined_info_block_1_1 = GCN_InfoShare_block(hidden_size)
        self.predefined_info_block_1_2 = GCN_InfoShare_block(hidden_size)

        self.predefined_info_block_2 = GCN_InfoShare_block(hidden_size)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)


    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1_1: torch.Tensor, # [N, C] 
        concept_matrix_1_2: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        # Layer 1
        p1_1_info_fore, p1_1_info_back = self.predefined_info_block_1_1(x_hidden, concept_matrix_1_1, market_value)
        p1_2_info_fore, p1_2_info_back = self.predefined_info_block_1_2(x_hidden, concept_matrix_1_2, market_value)

        x_hidden_dL1 = x_hidden - p1_1_info_back - p1_2_info_back

        # Layer 2
        p2_info_fore, p2_info_back = self.predefined_info_block_2(x_hidden_dL1, concept_matrix_2, market_value)

        x_hidden_dp1p2 = x_hidden_dL1 - p2_info_back

        # final layer
        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2))

        # output layer
        all_info_fore = p1_1_info_fore + p1_2_info_fore + p2_info_fore + i_info_fore

        return self.fc_out(all_info_fore)

        

class HIST_a11(FClipBase):
    """model 11 GCN[I] + GCN[b+] + GCN[r+]"""
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

        return self.fc_out(all_info_fore)

        
class HIST_a12(FClipBase):
    """model 10 GCN[Industry]GCN[Risk] + GCN[RetCorr]GCN[PrcCorr]"""
    def __init__(self, d_feat=6, hidden_size=64, clip=5):
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

        self.predefined_info_block_1_1 = GCN_InfoShare_block(hidden_size)
        self.predefined_info_block_1_2 = GCN_InfoShare_block(hidden_size)

        self.predefined_info_block_2_1 = GCN_InfoShare_block(hidden_size)
        self.predefined_info_block_2_2 = GCN_InfoShare_block(hidden_size)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)


    def forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1_1: torch.Tensor, # [N, C] 
        concept_matrix_1_2: torch.Tensor, # [N, C] 
        concept_matrix_2_1: torch.Tensor, # [N, C] 
        concept_matrix_2_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        # Layer 1
        p1_1_info_fore, p1_1_info_back = self.predefined_info_block_1_1(x_hidden, concept_matrix_1_1, market_value)
        p1_2_info_fore, p1_2_info_back = self.predefined_info_block_1_2(x_hidden, concept_matrix_1_2, market_value)

        x_hidden_dL1 = x_hidden - p1_1_info_back - p1_2_info_back

        # Layer 2
        p2_1_info_fore, p2_1_info_back = self.predefined_info_block_2_1(x_hidden_dL1, concept_matrix_2_1, market_value)
        p2_2_info_fore, p2_2_info_back = self.predefined_info_block_2_2(x_hidden_dL1, concept_matrix_2_2, market_value)

        x_hidden_dL1L2 = x_hidden_dL1 - p2_1_info_back - p2_2_info_back

        # final layer
        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dL1L2))

        # output layer
        all_info_fore = p1_1_info_fore + p1_2_info_fore + p2_1_info_fore + p2_2_info_fore + i_info_fore

        return self.fc_out(all_info_fore)
        

class HIST_FactorGCL_V2(FClipBase):
    """
    model FactorGCL V2
    GCN[Industry] + GCN[Risk] + GCN[corr] (与 HIST_a10 相同的三概念结构)
    dual output for contractive learning

    概念结构:
        概念 1_1 (v0)  : 一级行业
        概念 1_2 (v14) : 风险因子相关边
        概念 2   (v{n}) : 相关性边
    """
    def __init__(
        self, 
        d_feat=6, hidden_size=64, clip=5,
        ew1: bool = True, ew2: bool = True, ew3: bool = True,
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

        self.fut_encoder = nn.Sequential(
            nn.Linear(d_feat, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        for m in self.fut_encoder:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        # 与 HIST_a10 相同的三概念结构
        self.predefined_info_block_1_1 = GCN_InfoShare_block(hidden_size, ew=ew1)
        self.predefined_info_block_1_2 = GCN_InfoShare_block(hidden_size, ew=ew2)
        self.predefined_info_block_2 = GCN_InfoShare_block(hidden_size, ew=ew3)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)


    def real_forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1_1: torch.Tensor, # [N, C] 
        concept_matrix_1_2: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        # Layer 1: 行业 + 风险并行
        p1_1_info_fore, p1_1_info_back = self.predefined_info_block_1_1(x_hidden, concept_matrix_1_1, market_value)
        p1_2_info_fore, p1_2_info_back = self.predefined_info_block_1_2(x_hidden, concept_matrix_1_2, market_value)

        x_hidden_dL1 = x_hidden - p1_1_info_back - p1_2_info_back

        # Layer 2: 相关性边
        p2_info_fore, p2_info_back = self.predefined_info_block_2(x_hidden_dL1, concept_matrix_2, market_value)

        x_hidden_dp1p2 = x_hidden_dL1 - p2_info_back

        # final layer
        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2))

        # output layer
        all_info_fore = p1_1_info_fore + p1_2_info_fore + p2_info_fore + i_info_fore

        return self.fc_out(all_info_fore), i_info_fore

    
    def goof_forward(
        self,
        x_fut: torch.Tensor, # [N, F]
        concept_matrix_1_1: torch.Tensor, # [N, C] 
        concept_matrix_1_2: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.fut_encoder(self.fclip(x_fut))

        with torch.no_grad():
            # Layer 1: 行业 + 风险并行
            _, p1_1_info_back = self.predefined_info_block_1_1(x_hidden, concept_matrix_1_1, market_value)
            _, p1_2_info_back = self.predefined_info_block_1_2(x_hidden, concept_matrix_1_2, market_value)

            x_hidden_dL1 = x_hidden - p1_1_info_back - p1_2_info_back

            # Layer 2: 相关性边
            _, p2_info_back = self.predefined_info_block_2(x_hidden_dL1, concept_matrix_2, market_value)

            x_hidden_dp1p2 = x_hidden_dL1 - p2_info_back

            # final layer
            i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2))

        return i_info_fore


    def forward(
        self,
        concept_matrix_1_1: torch.Tensor, # [N, C] 
        concept_matrix_1_2: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
        x: torch.Tensor, # [N, F]
        x_fut: torch.Tensor | None = None, # [N, F]
        output: bool = False
    ):
        if output:
            batch_result, _ = self.real_forward(
                x, concept_matrix_1_1, concept_matrix_1_2, concept_matrix_2, market_value)
            return batch_result

        else:
            batch_result, real_indi_info = self.real_forward(
                x, concept_matrix_1_1, concept_matrix_1_2, concept_matrix_2, market_value)
            goof_indi_info = self.goof_forward(
                x_fut, concept_matrix_1_1, concept_matrix_1_2, concept_matrix_2, market_value)

            return batch_result, real_indi_info, goof_indi_info


class HIST_FactorGCL(FClipBase):
    """
    model FactorGCL 
    GCN + GCN
    dual output for contractive learning

    """
    def __init__(
        self, 
        d_feat=6, hidden_size=64, clip=5,
        ew1: bool = True, ew2: bool = True, 
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

        self.fut_encoder = nn.Sequential(
            nn.Linear(d_feat, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        for m in self.fut_encoder:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        self.predefined_info_block_1 = GCN_InfoShare_block(hidden_size, ew=ew1)
        self.predefined_info_block_2 = GCN_InfoShare_block(hidden_size, ew=ew2)

        self.fc_i_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_i_fore.weight)
        self.fc_out = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)


    def real_forward(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.encoder(self.fclip(x))

        # Layer 1
        p1_info_fore, p1_info_back = self.predefined_info_block_1(x_hidden, concept_matrix_1, market_value)
        x_hidden_dp1 = x_hidden - p1_info_back

        # Layer 2
        p2_info_fore, p2_info_back = self.predefined_info_block_2(x_hidden_dp1, concept_matrix_2, market_value)
        x_hidden_dp1p2 = x_hidden_dp1 - p2_info_back

        # final layer
        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2))

        # output layer
        all_info_fore = p1_info_fore + p2_info_fore + i_info_fore

        return self.fc_out(all_info_fore), i_info_fore

    
    def goof_forward(
        self,
        x_fut: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        x_hidden = self.fut_encoder(self.fclip(x_fut))

        with torch.no_grad():
            # Layer 1
            _, p1_info_back = self.predefined_info_block_1(x_hidden, concept_matrix_1, market_value)
            x_hidden_dp1 = x_hidden - p1_info_back

            # Layer 2
            _, p2_info_back = self.predefined_info_block_2(x_hidden_dp1, concept_matrix_2, market_value)
            x_hidden_dp1p2 = x_hidden_dp1 - p2_info_back

            # final layer
            i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2))

        return i_info_fore


    def forward(
        self,
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
        x: torch.Tensor, # [N, F]
        x_fut: torch.Tensor | None = None, # [N, F]
        output: bool = False
    ):
        if output:
            batch_result, _ = self.real_forward(
                x, concept_matrix_1, concept_matrix_2, market_value)
            return batch_result

        else:
            batch_result, real_indi_info = self.real_forward(
                x, concept_matrix_1, concept_matrix_2, market_value)
            goof_indi_info = self.goof_forward(
                x_fut, concept_matrix_1, concept_matrix_2, market_value)

            return batch_result, real_indi_info, goof_indi_info


class HIST_FactorGCL_V1(HIST_FactorGCL):

    def goof_forward(
        self,
        x_fut: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
    ) -> torch.Tensor:
        
        with torch.no_grad():
            x_hidden = self.encoder(self.fclip(x_fut))
            # Layer 1
            _, p1_info_back = self.predefined_info_block_1(x_hidden, concept_matrix_1, market_value)
            x_hidden_dp1 = x_hidden - p1_info_back

            # Layer 2
            _, p2_info_back = self.predefined_info_block_2(x_hidden_dp1, concept_matrix_2, market_value)
            x_hidden_dp1p2 = x_hidden_dp1 - p2_info_back

            # final layer
            i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2))

        return i_info_fore