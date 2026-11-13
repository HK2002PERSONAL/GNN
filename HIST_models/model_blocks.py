from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GCNConv, LayerNorm


class HIST_block_base(nn.Module):

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size

        self.fc_info = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_info.weight)
        self.fc_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_fore.weight)
        self.fc_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_back.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_s2t = torch.nn.Softmax(dim = 0)
        self.softmax_t2s = torch.nn.Softmax(dim = 1)


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



class HIST_PreDefinedConcept_block(HIST_block_base):

    def __init__(self, hidden_size: int = 64):
        """Predefined Concept Module"""
        super().__init__(hidden_size=hidden_size)
        
    # pylint:disable=W0221
    def forward(
        self, 
        x_hidden: torch.Tensor, # [N, F]
        concept_matrix: torch.Tensor, # [N, C]
        market_value: torch.Tensor # [N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算市值加权平均概念特征编码
        # [N] -> [N, 1] -> [N, C]
        market_value_matrix = market_value.reshape(market_value.shape[0], 1).repeat(1, concept_matrix.shape[1])
        # [N, C] * [N, C] = [N, C] 
        stock_to_concept = concept_matrix * market_value_matrix
        # [N, C] -> [N, 1] -> [N, C] 计算市值加权的分母
        stock_to_concept_sum = torch.sum(stock_to_concept, 0).reshape(1, -1).repeat(stock_to_concept.shape[0], 1)
        # [N, C] * [N, C] = [N, C] 
        stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)
        stock_to_concept_sum = stock_to_concept_sum + 1e-2 # avoid dividing 0
        # [N, C] / [N, C] = [N, C]
        stock_to_concept = stock_to_concept / stock_to_concept_sum # calulate weight
        # [C, N] @ [N, H] -> [C, H]
        c_hidden = torch.t(stock_to_concept).mm(x_hidden)
        # [C, H] -> [C', H]
        c_hidden = c_hidden[c_hidden.sum(1) != 0]
        # recalculate adjusted stock-concept relation
        # COS([N, H], [H, C']) -> [N, C']
        stock_to_concept = self.cal_cos_similarity(x_hidden, c_hidden)
        # [N, C'] -> [N, C']
        stock_to_concept = self.softmax_s2t(stock_to_concept) # calulate weight
        # [C', N] @ [N, H] -> [C', H]
        c_hidden = torch.t(stock_to_concept).mm(x_hidden)
        # COS([N, H], [C', H]) -> [N, C'] cosin similarity
        concept_to_stock = self.cal_cos_similarity(x_hidden, c_hidden) 
        concept_to_stock = self.softmax_t2s(concept_to_stock) # Softmax 权重化
        c_shared_info = concept_to_stock.mm(c_hidden)
        c_shared_info = self.fc_info(c_shared_info)
        # decode info
        fore_info = self.leaky_relu(self.fc_fore(c_shared_info))
        back_info = self.fc_back(c_shared_info)

        return fore_info, back_info


class HIST_HiddenConcept_block(HIST_block_base):

    def __init__(self, hidden_size: int = 64, K: int = 3):
        """Hidden Concept Module"""
        super().__init__(hidden_size=hidden_size)
        self.K = K

    # pylint:disable=W0221
    def forward(self, x_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 隐藏概念挖掘权重计算（x_hidden.device 兼容 CPU/CUDA）
        device = x_hidden.device
        # [N, H] -> [N, H]
        c_hidden = x_hidden
        # COS([N, H], [N, H]) -> [N, N]
        stock_to_concept = self.cal_cos_similarity(x_hidden, c_hidden)
        dim = stock_to_concept.shape[0] # N
        diag = stock_to_concept.diagonal(0).clone() # [N]
        # [N, N] -> [N, N] 对角线清零，去掉自环
        # stock_to_concept = stock_to_concept * (torch.ones(dim, dim) - torch.eye(dim)).to(device)
        stock_to_concept = stock_to_concept.fill_diagonal_(0)
        # 利用高级索引mask掉非TOPK的值
        row = torch.arange(dim).repeat_interleave(self.K).unsqueeze(0).to(device) # [N * K] 需要mask的部分的纵坐标
        column = torch.topk(stock_to_concept, self.K, dim = 1)[1].reshape(1, -1) # [N * K] 需要mask的部分的横坐标
        mask = torch.zeros([dim, dim], device=device) # [N, N]
        mask[row, column] = 1 # [N, N] 得到MASK
        # [N, N] -> [N, N] mask掉非TOPK的值
        stock_to_concept = stock_to_concept * mask
        # [N, N] -> [N, N] 把没被孤立的节点的自环加回来
        stock_to_concept = stock_to_concept + torch.diag_embed((stock_to_concept.sum(0) != 0).float() * diag)
        # [N, N] @ [N, H] -> [N, H]
        c_hidden = torch.t(x_hidden).mm(stock_to_concept).t()
        # 删掉没有任何股票的隐藏概念 [N, H] -> [M, H]
        c_hidden = c_hidden[c_hidden.sum(1) != 0]
        # 计算余弦相关度作为新的权重
        # COS([N, H], [M, H]) -> [N, M]
        h_concept_to_stock = self.cal_cos_similarity(x_hidden, c_hidden)
        h_concept_to_stock = self.softmax_t2s(h_concept_to_stock) # Softmax 权重化
        # [M, N] @ [N, H] -> [M, H]
        h_shared_info = h_concept_to_stock.mm(c_hidden)
        h_shared_info = self.fc_info(h_shared_info)
        # decode info
        fore_info = self.leaky_relu(self.fc_fore(h_shared_info))
        back_info = self.fc_back(h_shared_info)

        return fore_info, back_info


class GAT_InfoShare_block(HIST_block_base):
    """基于 GAT 的横截面信息共享模块（用于替换 Hidden Concept）

    forward 接口与 HIST_HiddenConcept_block 保持一致：
        输入 : x_hidden [N, H] 横截面内股票的隐藏表征
        输出 : (fore_info [N, H], back_info [N, H])
          - fore_info: 可预测的共享信息（经 LeakyReLU 激活）
          - back_info: 用于从个体信息中减去的共享信息（解耦用）

    设计要点（利用现代 attention 结构的优势）：
      1. 采用 GATv2Conv（ICLR'2022 GATv2）：attention 是节点特征的
         动态函数，表达能力强于一代 GAT 的静态 attention；
      2. 多头 attention：各头独立建模不同的横截面关系后拼接回原维度；
      3. 残差连接 + LayerNorm（Transformer 式结构），训练更稳定；
      4. 横截面图构建支持两种模式：
         - K=int （默认 3）: TopK 余弦相似度稀疏图，对齐 HIST 隐藏概念图，
           边数 O(N*K)，适合大截面（~5000 只股票）训练；
         - K=None: 全连接图，全量 attention（等价 self-attention，仅小截面使用）。
    """

    def __init__(
        self,
        hidden_size: int = 64,
        heads: int = 2,
        dropout: float = 0.0,
        K: Optional[int] = 3,
    ):
        super().__init__(hidden_size=hidden_size)
        # 多头 concat 后需保持输出维度=hidden_size 以支持残差，
        # 故 heads 必须整除 hidden_size；不满足时自动回退到可整除的最大头数
        orig_heads = heads
        while heads > 1 and hidden_size % heads != 0:
            heads -= 1
        self.heads = heads
        self.K = K

        # GATv2 多头 attention 堆叠
        self.conv = GATv2Conv(
            in_channels=hidden_size,
            out_channels=hidden_size // heads,
            heads=heads,
            concat=True,           # 多头拼接，保持输出维度为 hidden_size
            negative_slope=0.2,
            dropout=dropout,
            add_self_loops=False,  # 对齐 HIST：股票不关注自身
            share_weights=False,   # source/target 独立变换矩阵，表达能力更强
        )
        self.norm = nn.LayerNorm(hidden_size)

        self._fc_edge_cache = None  # 全连接图 edge_index 缓存 (N, device, edge_index)
        if orig_heads != heads:
            print(
                f"[GAT_InfoShare_block] heads {orig_heads}->{heads} "
                f"(需整除 hidden_size={hidden_size})"
            )

    def _build_edge_index(self, x_hidden: torch.Tensor) -> torch.Tensor:
        """构建横截面图的 edge_index [2, E]，消息方向 source -> target。"""
        device = x_hidden.device
        N = x_hidden.shape[0]

        if self.K is not None:
            # TopK 余弦相似度图（对齐 HIST_HiddenConcept_block 的建图方式）
            sim = self.cal_cos_similarity(x_hidden, x_hidden)  # [N, N]
            # sim = sim * (1.0 - torch.eye(N, device=device))    # 对角线清零，去掉自环
            sim.fill_diagonal_(0) 
            _, topk_idx = torch.topk(sim, min(self.K, N - 1), dim=1)  # [N, K']
            K = topk_idx.shape[1]
            target = torch.arange(N, device=device).unsqueeze(1).expand(-1, K).reshape(-1)
            source = topk_idx.reshape(-1)
            return torch.stack([source, target], dim=0)

        # 全连接图（去除自环），结果缓存避免重复构建
        cache = self._fc_edge_cache
        if cache is not None and cache[0] == N and cache[1] == device:
            return cache[2]
        idx = torch.arange(N, device=device)
        target = idx.repeat_interleave(N)
        source = idx.repeat(N)
        mask = source != target
        edge_index = torch.stack([source[mask], target[mask]], dim=0)
        self._fc_edge_cache = (N, device, edge_index)
        return edge_index

    # pylint:disable=W0221
    def forward(self, x_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = self._build_edge_index(x_hidden)

        # GAT 多头 attention 横截面信息聚合（残差 + LayerNorm）
        h = x_hidden
        h = self.norm(h + self.conv(h, edge_index))

        # 解码共享信息，输出结构与 HIST 保持一致
        shared_info = self.fc_info(h)
        fore_info = self.leaky_relu(self.fc_fore(shared_info))
        back_info = self.fc_back(shared_info)

        return fore_info, back_info


class GCN_InfoShare_block(HIST_block_base):
    """基于 GCN 的横截面信息共享模块（用于替换 Predefined Concept）

    forward 接口与 HIST_PreDefinedConcept_block 保持一致：
        输入 : x_hidden [N, H] 股票隐藏表征
               concept_matrix [N, C] 股票-概念隶属矩阵（0/1）
               market_value [N] 股票市值
        输出 : (fore_info [N, H], back_info [N, H])
          - fore_info: 可预测的概念共享信息（经 LeakyReLU 激活）
          - back_info: 用于从个体信息中减去的共享信息（解耦用）

    设计要点：
      1. 将 HIST 手写的"市值加权 + 余弦重校准"两段聚合，替换为
         股票-概念二部图上的现成 GCNConv（Kipf & Welling'17）传播：
         第 1 层 stock -> concept 聚合概念表征，第 2 层 concept -> stock
         将概念共享信息回传股票，天然实现横截面信息共享；
      2. 边权使用归一化市值，对齐 HIST 的市值加权语义（大市值股主导概念）；
      3. 残差连接 + LayerNorm（Project GAT / GraphGPS 式深层 GCN 稳定结构）；
      4. GCNConv 对称归一化自带度缩放，概念节点度即成分股数量，
         无需像 HIST 一样手动计算 softmax 权重。
    """

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
        """构建股票-概念二部图（stock 节点 0..N-1，concept 节点 N..N+C'-1）。

        return: edge_index [2, E], edge_weight [E], n_concepts
        """
        N = concept_matrix.shape[0]
        # 剔除没有任何成分股的空概念（对齐 HIST 的 c_hidden.sum(1)!=0 处理）
        valid = concept_matrix.sum(0) != 0          # [C]
        cm = concept_matrix[:, valid]               # [N, C']
        n_concepts = int(cm.shape[1])

        stock_idx, concept_idx = torch.nonzero(cm, as_tuple=True)
        concept_idx = concept_idx + N               # concept 节点编号偏移
        edge_s2c = torch.stack([stock_idx, concept_idx], dim=0)     # [2, E]
        # 双向边：stock->concept 与 concept->stock，使 GCN 在二部图上
        # 一次消息传递即可完成"聚合-分发"，两层堆叠形成两跳信息共享
        edge_index = torch.cat([edge_s2c, edge_s2c.flip(0)], dim=1)  # [2, 2E]

        # 归一化市值作为边权（均值约1，保持特征尺度稳定）
        mv = market_value.detach().clamp_min(0.0)
        mv = mv / (mv.sum() + 1e-8) * N
        # 双向边：正向(s2c)与反向(c2s)使用相同的市值权重，需复制一份对齐 2E 长度
        edge_weight = mv[stock_idx].repeat(2)

        return edge_index, edge_weight, n_concepts

    # pylint:disable=W0221
    def forward(
        self,
        x_hidden: torch.Tensor,     # [N, H]
        concept_matrix: torch.Tensor,  # [N, C]
        market_value: torch.Tensor,    # [N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = x_hidden.shape[0]

        edge_index, edge_weight, n_concepts = self._build_bipartite_graph(
            concept_matrix, market_value)

        if n_concepts == 0 or edge_index.numel() == 0:
            # 无可用概念 -> 无共享信息（与 HIST 语义一致，且数值安全）
            zero = torch.zeros_like(x_hidden)
            return zero, zero

        # 全图节点特征：stock 段用隐藏表征，concept 段置零（由聚合填充）
        x_all = torch.cat(
            [x_hidden, x_hidden.new_zeros(n_concepts, self.hidden_size)], dim=0
        )

        # 第一跳：stock -> concept 聚合（残差保留个体信息）+ LayerNorm
        h = self.conv1(x_all, edge_index, edge_weight) if self.ew else self.conv1(x_all, edge_index)
        h = self.norm1(h + x_all)
        h = self.drop(self.leaky_relu(h))

        # 第二跳：concept -> stock 分发共享信息
        h = self.conv2(h, edge_index, edge_weight) if self.ew else self.conv2(h, edge_index)
        x_shared = self.norm2(h[:N] + x_hidden)  # 取 stock 段并残差

        # 解码共享信息，输出结构与 HIST 保持一致
        shared_info = self.fc_info(x_shared)
        fore_info = self.leaky_relu(self.fc_fore(shared_info))
        back_info = self.fc_back(shared_info)

        return fore_info, back_info


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
        x_shared_info = self.fc_info(x_shared)
        # decode info
        fore_info = self.leaky_relu(self.fc_fore(x_shared_info))
        back_info = self.fc_back(x_shared_info)

        return fore_info, back_info


class RGAT_InfoShare_block(GAT_InfoShare_block):

    # pylint:disable=W0221
    def forward(
        self,
        x_hidden: torch.Tensor, # [N, H]
        risk_feature: torch.Tensor, # [N, R]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        edge_index = self._build_edge_index(risk_feature)

        # GAT 多头 attention 横截面信息聚合（残差 + LayerNorm）
        h = x_hidden
        h = self.norm(h + self.conv(h, edge_index))

        # 解码共享信息，输出结构与 HIST 保持一致
        shared_info = self.fc_info(h)
        fore_info = self.leaky_relu(self.fc_fore(shared_info))
        back_info = self.fc_back(shared_info)

        return fore_info, back_info







