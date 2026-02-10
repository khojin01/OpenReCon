import torch
import numpy as np
import sys
import os
import pickle as pk
import scipy.sparse as ss
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

src_path = os.path.join(current_dir, 'src')
signet_path = os.path.join(src_path, 'SigNet')
sys.path.append(signet_path)

try:
    from signet.cluster import Cluster
    from signet.utils import sqrtinvdiag
    from src.metrics import calculate_acc, compute_modularity, compute_conductance, calculate_unhappy_ratio
    from src.initial_clustering import run_initial_clustering
    from src.pipeline import run_full_pipeline
    from param_parser import parse_args
except ImportError as e:
    print(f"Import error: {e}")
    print("Current working directory:", os.getcwd())
    print("Python path:", sys.path)
    sys.exit(1)

def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print(f"ReCon Pipeline")
    print(f"{'='*70}")
    print(f"Data: {args.data_path}")
    print(f"Initial Method: {args.initial_method}")
    print(f"Pipeline: {args.iterations} iterations")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    print("[Step 1] Loading data...")
    with open(args.data_path, "rb") as f:
        data = pk.load(f)
    
    A_p, A_n = data['A_p'], data['A_n']
    feat_L = data.get('feat_L')
    labels = data.get('labels') or data.get('y')
    
    if feat_L is None:
        feat_L = ss.eye(A_p.shape[0])
    
    if args.K is None:
        if labels is not None:
            true_k = len(np.unique(labels))
        else:
            true_k = 6
    else:
        true_k = args.K
    
    print(f"  Nodes: {A_p.shape[0]}, +edges: {A_p.nnz}, -edges: {A_n.nnz}")
    print(f"  K = {true_k}\n")
    
    print(f"[Step 2] Initial clustering ({args.initial_method})...")
    cluster_model = Cluster((A_p, A_n))
    
    initial_clusters = run_initial_clustering(
        args.initial_method, A_p, A_n, feat_L, true_k, device, args, cluster_model
    )
    
    print(f"  Initial clustering complete: {len(np.unique(initial_clusters))} clusters")
    
    cluster_counts = Counter(initial_clusters)
    sorted_counts = dict(sorted(cluster_counts.items()))
    print(f"  Initial cluster distribution:")
    for cluster_id, count in sorted_counts.items():
        print(f"    Cluster {cluster_id}: {count} nodes ({count/len(initial_clusters)*100:.2f}%)")
    print()
    
    print("[Step 3] Running pipeline...")
    final_clusters, final_embeddings = run_full_pipeline(args, A_p, A_n, feat_L, initial_clusters, device)
    print(f"  Pipeline complete: {len(np.unique(final_clusters))} clusters")
    if final_embeddings is not None:
        print(f"  Final embeddings shape: {final_embeddings.shape}")
    
    cluster_counts = Counter(final_clusters)
    sorted_counts = dict(sorted(cluster_counts.items()))
    print(f"  final cluster distribution:")
    for cluster_id, count in sorted_counts.items():
        print(f"    Cluster {cluster_id}: {count} nodes ({count/len(final_clusters)*100:.2f}%)")
    print()
    
    print("\n[Step 4] Computing performance metrics...")
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print(f"\n{'='*70}")
    print(f"PERFORMANCE METRICS")
    print(f"{'='*70}")

    initial_modularity = compute_modularity(initial_clusters, A_p, A_n)
    initial_conductance = compute_conductance(initial_clusters, A_p, A_n)
    initial_unhappy = calculate_unhappy_ratio(A_p, A_n, initial_clusters)
    if labels is not None:
        initial_ari = adjusted_rand_score(labels, initial_clusters)
        initial_nmi = normalized_mutual_info_score(labels, initial_clusters)
        initial_acc = calculate_acc(labels, initial_clusters)
    else:
        initial_ari = 0.0
        initial_nmi = 0.0
        initial_acc = 0.0
    print(f"Initial (Before Pipeline): (ACC: {initial_acc:.4f}, ARI: {initial_ari:.4f}, NMI: {initial_nmi:.4f}, Modularity: {initial_modularity:.4f}, Conductance: {initial_conductance:.4f}, Unhappy: {initial_unhappy:.4f})")

    final_modularity = compute_modularity(final_clusters, A_p, A_n)
    final_conductance = compute_conductance(final_clusters, A_p, A_n)
    final_unhappy = calculate_unhappy_ratio(A_p, A_n, final_clusters)
    if labels is not None:
        final_ari = adjusted_rand_score(labels, final_clusters)
        final_nmi = normalized_mutual_info_score(labels, final_clusters)
        final_acc = calculate_acc(labels, final_clusters)
    else:
        final_ari = 0.0
        final_nmi = 0.0
        final_acc = 0.0
    print(f"Final (After Pipeline):   (ACC: {final_acc:.4f}, ARI: {final_ari:.4f}, NMI: {final_nmi:.4f}, Modularity: {final_modularity:.4f}, Conductance: {final_conductance:.4f}, Unhappy: {final_unhappy:.4f})")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
