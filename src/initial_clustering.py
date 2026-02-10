import os
import sys
import numpy as np
import torch
from sklearn.cluster import KMeans

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

src_dir = current_dir
sssnet_path = os.path.join(src_dir, 'SSSNET')
dsgc_path = os.path.join(src_dir, 'DSGC')
if sssnet_path not in sys.path:
    sys.path.append(sssnet_path)
if dsgc_path not in sys.path:
    sys.path.append(dsgc_path)

def run_initial_clustering(method, A_p, A_n, feat_L, k, device, args, cluster_model):
    if method == 'SPONGE':
        return cluster_model.SPONGE(k=k)
    elif method == 'SPONGE_sym':
        return cluster_model.SPONGE_sym(k=k)
    elif method == 'spectral_cluster_laplacian':
        return cluster_model.spectral_cluster_laplacian(k=k, normalisation=args.normalisation)
    elif method == 'spectral_cluster_adjacency':
        return cluster_model.spectral_cluster_adjacency(k=k, normalisation=args.normalisation)
    elif method == 'spectral_cluster_adjacency_reg':
        return cluster_model.spectral_cluster_adjacency_reg(k=k, normalisation=args.normalisation)
    elif method == 'spectral_cluster_bnc':
        return cluster_model.spectral_cluster_bnc(k=k, normalisation=args.normalisation)
    elif method == 'spectral_cluster_bethe_hessian':
        return cluster_model.spectral_cluster_bethe_hessian(k=k, normalisation=args.normalisation)
    elif method == 'geproblem_adjacency':
        return cluster_model.geproblem_adjacency(k=k, normalisation=args.normalisation)
    elif method == 'geproblem_laplacian':
        return cluster_model.geproblem_laplacian(k=k, normalisation=args.normalisation)
    elif method == 'SDP_cluster':
        return cluster_model.SDP_cluster(k=k, normalisation=args.normalisation, solver=args.solver)
    elif method == 'sssnet':
        import sssnet_native
        return sssnet_native.run_sssnet_clustering_direct(A_p, A_n, feat_L, k, device, epochs=500, data_path=args.data_path, seed=args.seed)
    elif method == 'dsgc':
        import dsgc_native
        return dsgc_native.run_dsgc_clustering(
            A_p, A_n, feat_L, k, device,
            epochs=args.dsgc_epochs,
            data_path=args.data_path,
            seed=args.seed,
            hidden=args.dsgc_hidden,
            dropout=args.dsgc_dropout,
            hop=args.dsgc_hop,
            m_p=args.dsgc_m_p,
            m_n=args.dsgc_m_n,
            tau=args.dsgc_tau,
            delta_p=args.dsgc_delta_p,
            delta_n=args.dsgc_delta_n,
            eta=args.dsgc_eta,
            apply_vsr=args.dsgc_apply_vsr,
            lr=args.dsgc_lr,
            weight_decay=args.dsgc_weight_decay,
            feature_type=args.dsgc_feature_type,
            pbnc_lambda=args.dsgc_pbnc_lambda,
            pbnc_loss=args.dsgc_pbnc_loss,
            directed=args.dsgc_directed,
            dense=args.dsgc_dense,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
