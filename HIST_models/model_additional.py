import torch
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm

# pylint:disable=W0223
class HLGCNConv(MessagePassing):
    """高通图卷积层（异配边专用），接口与 GCNConv 一致。

    传播规则:  H' = L̂_sym X W = (I - Â_sym) X W

    与 GCNConv 的唯一区别在符号：
      GCNConv (低通):  H' = Â_sym X W      节点自身 + 邻居聚合  → 抹平差异
      HLGCNConv(高通): H' = X - Â_sym X    节点自身 - 邻居聚合  → 放大差异

    参数与 GCNConv 相同：
        in_channels / out_channels: 输入/输出维度
        improved:   若为 True, 自环权重取 2 (对应 Â 的 Kipf improved 变体)
        cached:     是否缓存归一化后的边权 (transductive 固定图建议开)
        add_self_loops: 是否加自环 (默认 True, 与 GCNConv 一致)
        normalize:  是否做对称归一化 (默认 True)
        bias:       是否加偏置
    """

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = False, normalize: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    # pylint:disable=W0221
    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        # 1. 归一化：与 GCNConv 完全相同的 gcn_norm，得到 Â_sym 的边权
        if isinstance(edge_index, Tensor):
            if self.normalize:
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        improved=self.improved,
                        add_self_loops=self.add_self_loops,
                        flow=self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache
            # 2. 低通聚合 agg = Â_sym X
            agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        elif isinstance(edge_index, SparseTensor):
            if self.normalize:
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        improved=self.improved,
                        add_self_loops=self.add_self_loops,
                        flow=self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
            agg = spmm(edge_index, x, reduce=self.aggr)
        else:
            raise TypeError(f'Unsupported edge_index type: {type(edge_index)}')

        # 3. 关键一步：高通 = 自身 - 低通聚合,  即 (I - Â_sym) X
        out = x - agg

        # 4. 线性变换 + 偏置
        out = self.lin(out)
        if self.bias is not None:
            out = out + self.bias
        return out

    # pylint:disable=W0221
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j