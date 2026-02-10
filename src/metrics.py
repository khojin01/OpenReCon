import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size

def compute_modularity(clusters, A_p, A_n):
    A_p = A_p.tocsr()
    A_n = A_n.tocsr()

    m_plus = float(A_p.sum()) / 2.0
    m_minus = float(A_n.sum()) / 2.0

    if m_plus == 0 and m_minus == 0:
        return 0.0

    Q = 0.0
    communities = np.unique(clusters)

    if m_plus > 0:
        Qp = 0.0
        for com in communities:
            nodes = np.where(clusters == com)[0]
            if nodes.size == 0:
                continue
            e_c_plus = A_p[nodes, :][:, nodes].sum()
            d_c_plus = A_p[nodes, :].sum()
            Qp += (e_c_plus - (d_c_plus ** 2) / (2.0 * m_plus))
        Q += Qp / (2.0 * m_plus)

    if m_minus > 0:
        Qm = 0.0
        for com in communities:
            nodes = np.where(clusters == com)[0]
            if nodes.size == 0:
                continue
            e_c_minus = A_n[nodes, :][:, nodes].sum()
            d_c_minus = A_n[nodes, :].sum()
            Qm += (e_c_minus - (d_c_minus ** 2) / (2.0 * m_minus))
        Q -= Qm / (2.0 * m_minus)

    return float(Q)


def compute_conductance(clusters, A_p, A_n):
    try:
        import scipy.sparse as sp
        
        if not sp.issparse(A_p):
            A_p = sp.csr_matrix(A_p)
        if not sp.issparse(A_n):
            A_n = sp.csr_matrix(A_n)
        
        labels = np.asarray(clusters)
        degrees_p = np.array(A_p.sum(axis=1)).flatten()
        degrees_n = np.array(A_n.sum(axis=1)).flatten()
        total_degrees = degrees_p + degrees_n
        total_volume = total_degrees.sum()
        
        conductances = []
        
        for cluster_id in np.unique(labels):
            cluster_mask = (labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            vol_S = total_degrees[cluster_mask].sum()
            vol_complement = total_volume - vol_S
            min_vol = min(vol_S, vol_complement)
            if min_vol == 0:
                continue
            
            cut = 0.0
            
            for i in cluster_indices:
                row_p = A_p.getrow(i)
                for j_idx, j in enumerate(row_p.indices):
                    if not cluster_mask[j]:
                        cut += row_p.data[j_idx]
            
            for i in cluster_indices:
                row_n = A_n.getrow(i)
                for j_idx, j in enumerate(row_n.indices):
                    if cluster_mask[j]:
                        cut += row_n.data[j_idx]
            
            conductance = cut / min_vol
            conductances.append(conductance)
        
        return float(np.mean(conductances)) if conductances else 0.0
    
    except Exception as e:
        print(f"Error calculating conductance: {e}")
        return 0.0

def calculate_unhappy_ratio(A_p, A_n, clustering):
    unhappy_count = 0
    total_edges = 0
    
    A_p_coo = A_p.tocoo()
    for i, j in zip(A_p_coo.row, A_p_coo.col):
        if i != j:
            total_edges += 1
            if clustering[i] != clustering[j]:
                unhappy_count += 1
    
    A_n_coo = A_n.tocoo()
    for i, j in zip(A_n_coo.row, A_n_coo.col):
        if i != j:
            total_edges += 1
            if clustering[i] == clustering[j]:
                unhappy_count += 1
    
    return unhappy_count / total_edges if total_edges > 0 else 0.0
