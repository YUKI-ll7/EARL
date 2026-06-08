import torch
import torch.nn as nn
import torch.nn.functional as F
from .afnonet import Mlp
from .afnonet import PatchEmbed

def batched_spmm(indices, values, shape, h):
    """
    批处理稀疏-稠密矩阵乘法（所有批次稀疏结构相同）
    Args:
        indices: (2, nnz), 所有批次共享的稀疏矩阵坐标
        values: (B, nnz), 各批次稀疏矩阵的非零值
        b:      (B, N, C), 稠密矩阵
    Returns:
        (B, N, C): 乘法结果
    """

    B, N, C = h.shape
    dv = h.device
    row_offset = torch.arange(B, device=dv) * shape[0]  # (B,)
    col_offset = torch.arange(B, device=dv) * shape[1]  # (B,)
    
    # 扩展 indices 到块对角格式
    diag_indices_row = indices[0].unsqueeze(0) + row_offset.unsqueeze(1)  # (B, nnz)
    diag_indices_col = indices[1].unsqueeze(0) + col_offset.unsqueeze(1)  # (B, nnz)

    all_indices = torch.stack([
        diag_indices_row.reshape(-1),  # (B*nnz,)
        diag_indices_col.reshape(-1)   # (B*nnz,)
    ], dim=0)  # (2, B*nnz)

    all_values = values.reshape(-1)  # (B*nnz,)

    big_sparse = torch.sparse_coo_tensor(
        all_indices,
        all_values,
        size=(B*shape[0], B*shape[1])
    )

    b = torch.ones((B*shape[0],1), device=dv)
    h = h.reshape(-1, C)  # (B*N, C)

    e_rowsum = torch.sparse.mm(big_sparse.t(), b) 
    h_prime = torch.sparse.mm(big_sparse.t(), h) 
     # (B*N, C)
    h_prime = h_prime.reshape(B, shape[1], C)
    h_prime = h_prime.div(e_rowsum.reshape(B, shape[1], 1))
    
    return h_prime


   
class BatchGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(BatchGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.a = nn.Parameter(torch.zeros(size=(2*out_features+8, 1)))
        # 改用更小的初始化范围
        nn.init.normal_(self.a.data, mean=0, std=0.01)

        # self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.leakyrelu = nn.GELU()
        
        
    def forward(self, mesh, grid, edge, edge_index):

        _, M, _ = mesh.size()
        B, N, _ = grid.size()

        h = torch.cat([mesh, grid], dim=-2)  # [B, M+N, out_features]
        edge = edge.unsqueeze(0).expand(B, -1, -1, -1)  # [B, M+N, N, out_features]
        a_input = torch.cat([h[:, edge_index[0, :], :], mesh[:, edge_index[1, :], :], edge[:, edge_index[0, :], edge_index[1, :]]], dim=-1)  # [B, N, N, 2*out_features]
        edge_value = torch.exp(-self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1)))#注意力中的分子        
        h_prime = batched_spmm(edge_index, edge_value, torch.Size([M+N, M]), h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class BatchGAT(nn.Module):
    def __init__(self, img_size, patch_size, grid_feat, mesh_feat, grid_embed_dim, dropout, alpha, nheads):
        super(BatchGAT, self).__init__()
        self.dropout = dropout
        
        self.grid_embed = Mlp(in_features=grid_feat, hidden_features=grid_embed_dim, out_features=grid_embed_dim, act_layer=nn.GELU)
        self.mesh_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=mesh_feat, embed_dim=grid_embed_dim)
        self.mlp_mesh = nn.Linear(grid_embed_dim, grid_embed_dim)
        self.mlp_edge = Mlp(in_features=2, hidden_features=4, out_features=8, act_layer=nn.GELU)
        

        self.attentions = [BatchGraphAttentionLayer(grid_feat, grid_embed_dim, dropout=dropout, alpha=alpha, concat=True) 
                          for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = nn.Linear(grid_embed_dim * nheads, grid_embed_dim)

    def forward(self, mesh, grid, edge):
        # x shape: [B, N, nfeat]
        # adj shape: [B, N, N]
        
        mesh = self.mlp_mesh(self.mesh_embed(mesh))
        grid = self.grid_embed(grid)
        adj = edge[0]
        edge_index = adj.nonzero().t() 
        
        edge = self.mlp_edge(edge[1:].permute(1, 2, 0))

        x = torch.cat([att(mesh, grid, edge, edge_index) for att in self.attentions], dim=2)  # [B, N, nhid * nheads]
        x = F.dropout(x, self.dropout, training=self.training)
        x =  self.out_att(x)  # [B, N, nclass]

        return x  # [B, N, nclass]

# B = 2  # batch size
# N = 99  # number of nodes

# mesh_feat = 5  # 特征维度
# grid_feat = 5  # 特征维度

# # 创建模型
# model = BatchGAT(img_size = (256,256), patch_size = (16,16),grid_feat = grid_feat, mesh_feat=mesh_feat, grid_embed_dim=96, dropout=0.0, alpha=0.1, nheads=8)

# # 创建随机输入数据和邻接矩阵
# mesh = torch.randn(B, mesh_feat, 256, 256)  # 输入特征 [B, N, C]
# grid = torch.randn(B, N, grid_feat)  # 输入特征 [B, N, C]
# edge = torch.randn(3, 99+256, 256) # 邻接矩阵 [B, N, N]



# # 前向传播
# output = model(mesh, grid, edge)  # 输出 [B, N, nclass]
# print(output.shape)