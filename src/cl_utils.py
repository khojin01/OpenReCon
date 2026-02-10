import random
from itertools import permutations

import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def drop_features(x: Tensor, p: float):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, cluster_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if cluster_attr is None else cluster_attr[mask]


def drop_incidence(cluster_index: Tensor, p: float = 0.2):
    if p == 0.0:
        return cluster_index
    
    row, col = cluster_index
    mask = torch.rand(row.size(0), device=cluster_index.device) >= p
    
    row, col, _ = filter_incidence(row, col, None, mask)
    cluster_index = torch.stack([row, col], dim=0)
    return cluster_index


def drop_nodes(cluster_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return cluster_index

    drop_mask = torch.rand(num_nodes, device=cluster_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(cluster_index, \
        cluster_index.new_ones((cluster_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[drop_idx, :] = 0
    cluster_index = H.to_sparse().indices()

    return cluster_index


def drop_clusters(cluster_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return cluster_index

    drop_mask = torch.rand(num_edges, device=cluster_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(cluster_index, \
        cluster_index.new_ones((cluster_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[:, drop_idx] = 0
    cluster_index = H.to_sparse().indices()

    return cluster_index


def valid_node_edge_mask(cluster_index: Tensor, num_nodes: int, num_edges: int):
    ones = cluster_index.new_ones(cluster_index.shape[1])
    Dn = scatter_add(ones, cluster_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, cluster_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def common_node_edge_mask(cluster_indexs: list[Tensor], num_nodes: int, num_edges: int):
    cluster_weight = cluster_indexs[0].new_ones(num_edges)
    node_mask = cluster_indexs[0].new_ones((num_nodes,)).to(torch.bool)
    edge_mask = cluster_indexs[0].new_ones((num_edges,)).to(torch.bool)

    for index in cluster_indexs:
        Dn = scatter_add(cluster_weight[index[1]], index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(index.new_ones(index.shape[1]), index[1], dim=0, dim_size=num_edges)
        node_mask &= Dn != 0
        edge_mask &= De != 0
    return node_mask, edge_mask


def cluster_index_masking(cluster_index, num_nodes, num_edges, node_mask, edge_mask):
    if node_mask is None and edge_mask is None:
        return cluster_index

    H = torch.sparse_coo_tensor(cluster_index, \
        cluster_index.new_ones((cluster_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_cluster_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_cluster_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_cluster_index = H[node_mask].to_sparse().indices()
    return masked_cluster_index


def clique_expansion(cluster_index: Tensor):
    edge_set = set(cluster_index[1].tolist())
    adjacency_matrix = []
    for edge in edge_set:
        mask = cluster_index[1] == edge
        nodes = cluster_index[:, mask][0].tolist()
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)
    
    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    return adjacency_matrix.to(cluster_index.device)
