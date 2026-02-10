import argparse

def parse_args():
    p = argparse.ArgumentParser(description='ReCon: Robust Clustering for Signed Networks')
    
    # === Data Parameters ===
    p.add_argument("--data-path", type=str, required=True,
                   help="Path to data file")
    p.add_argument("--output-path", type=str, default="./output",
                   help="Output directory path")
    p.add_argument("--dataset", type=str, default="SP1500",
                   choices=["SP1500", "synthetic", "real"],
                   help="Dataset type")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--save-predictions", action="store_true", default=False,
                   help="Save prediction labels to .npy file")
    
    # === Device Parameters ===
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                   help="Device to use (auto: auto-select, cpu: force CPU, cuda: use GPU)")
    p.add_argument("--debug", action="store_true", default=False,
                   help="Debug mode (verbose output)")

    
    # === Clustering Parameters ===
    p.add_argument("--K", type=int, default=None,
                   help="Number of clusters (None for auto-detection from labels)")
    p.add_argument("--initial-clusterer", type=str, default="signet",
                   choices=["signet"],
                   help="Initial clustering algorithm")
    p.add_argument("--initial-method", type=str, default="SPONGE",
                   choices=["SPONGE", "SPONGE_sym", "spectral_cluster_adjacency", 
                           "spectral_cluster_adjacency_reg", "spectral_cluster_bnc",
                           "spectral_cluster_laplacian", "spectral_cluster_bethe_hessian",
                           "geproblem_adjacency", "geproblem_laplacian", "SDP_cluster",
                           "sssnet", "sigat", "sigmanet", "signet", 
                           "sdgnn", "sgcl", "signed_louvain", "dsgc"],
                   help="Initial clustering method")
    p.add_argument("--normalisation", type=str, default="sym",
                   choices=["none", "sym", "rw", "sym_sep", "rw_sep", "neg", 
                           "additive", "multiplicative"],
                   help="Normalisation method for SigNet clustering")
    p.add_argument("--solver", type=str, default="BM_proj_grad",
                   help="Solver for SDP_cluster")
    p.add_argument("--iterations", type=int, default=3,
                   help="Number of main loop iterations")
    
    # === Cluster Construction Parameters ===
    p.add_argument("--max-samples-per-edge", type=int, default=100,
                   help="Maximum samples per edge")
    p.add_argument("--lambda-purge", type=float, default=0.1,
                   help="Lambda purge for cluster construction (friend vs enemy ratio)")
    p.add_argument("--s-min-import", type=float, default=0.3,
                   help="Minimum triangle participation ratio for import")
    p.add_argument("--p-min-import", type=float, default=0.8,
                   help="Positive triangle density threshold for import")
    p.add_argument("--top-m-ratio", type=float, default=0.5,
                   help="Top M ratio")
    p.add_argument("--max-cluster-membership", type=int, default=10,
                   help="Maximum cluster membership")
    p.add_argument("--max-overlap-ratio", type=float, default=0.5,
                   help="Maximum overlap ratio")
    p.add_argument("--min-cluster-size", type=int, default=5,
                   help="Minimum cluster size")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Number of CPU cores for parallel processing (-1: all cores)")
    
    # === DSGC Parameters ===
    p.add_argument("--dsgc-epochs", type=int, default=700,
                   help="Number of epochs for DSGC")
    p.add_argument("--dsgc-hidden", type=int, default=32,
                   help="Hidden dimension for DSGC")
    p.add_argument("--dsgc-dropout", type=float, default=0.5,
                   help="Dropout rate for DSGC")
    p.add_argument("--dsgc-hop", type=int, default=2,
                   help="Number of hops for DSGC")
    p.add_argument("--dsgc-m-p", type=int, default=3,
                   help="M_p parameter for DSGC")
    p.add_argument("--dsgc-m-n", type=int, default=2,
                   help="M_n parameter for DSGC")
    p.add_argument("--dsgc-tau", type=float, default=0.0,
                   help="Tau parameter for DSGC")
    p.add_argument("--dsgc-lr", type=float, default=0.01,
                   help="Learning rate for DSGC")
    p.add_argument("--dsgc-weight-decay", type=float, default=5e-4,
                   help="Weight decay for DSGC")
    p.add_argument("--dsgc-feature-type", type=str, default="A_reg",
                   choices=["A_reg", "L", "given"],
                   help="Feature type for DSGC")
    p.add_argument("--dsgc-pbnc-lambda", type=float, default=0.03,
                   help="PBNC lambda for DSGC")
    p.add_argument("--dsgc-pbnc-loss", type=float, default=1.0,
                   help="PBNC loss weight for DSGC")
    p.add_argument("--dsgc-directed", action="store_true",
                   help="Use directed graph for DSGC")
    p.add_argument("--dsgc-dense", action="store_true",
                   help="Use dense representation for DSGC")
    p.add_argument("--dsgc-delta-p", type=int, default=1,
                   help="Delta_p parameter for DSGC")
    p.add_argument("--dsgc-delta-n", type=int, default=1,
                   help="Delta_n parameter for DSGC")
    p.add_argument("--dsgc-eta", type=float, default=0.0,
                   help="Eta parameter for DSGC")
    p.add_argument("--dsgc-apply-vsr", action="store_true",
                   help="Apply VSR for DSGC")
    
    
    # === Refinement Algorithm Selection ===
    p.add_argument("--refinement-method", type=str, default="soft_label_propagation",
                   choices=["soft_label_propagation", "edge_flipping"],
                   help="Cluster refinement algorithm")
    
    # === Soft Label Propagation Refinement Parameters ===
    p.add_argument("--refine-alpha", type=float, default=0.8,
                   help="Graph structure weight (alpha)")
    p.add_argument("--refine-beta", type=float, default=0.2,
                   help="Embedding similarity weight (beta)")
    p.add_argument("--refine-neg-weight", type=float, default=3.0,
                   help="Negative edge weight")
    p.add_argument("--refine-temp", type=float, default=1.0,
                   help="Initial temperature parameter for soft label propagation")
    p.add_argument("--refine-temp-decay", type=float, default=0.8,
                   help="Temperature decay rate for soft label propagation")
    p.add_argument("--refine-stochastic", action="store_true", default=False,
                   help="Use stochastic label update")
    p.add_argument("--refine-debug", action="store_true", default=False,
                   help="Refinement debug mode")
    
    # === Edge Flipping Refinement Parameters ===
    p.add_argument("--refine-unbalanced-threshold", type=float, default=0.5,
                   help="Unbalanced triangle threshold for edge flipping (0-1, skip if < 0)")
    
    # === Re-clustering Parameters ===
    p.add_argument("--recluster-method", type=str, default="kmeans",
                   choices=["kmeans", "agglomerative", "spectral", "spectral_cosine", "spectral_rbf", "signed_kmeans"],
                   help="Re-clustering algorithm")
    
    # K-Means Parameters
    p.add_argument("--recluster-n-init", type=int, default=10,
                   help="K-Means n_init parameter")
    p.add_argument("--recluster-max-iter", type=int, default=300,
                   help="K-Means/GMM max_iter parameter")
    
    # Agglomerative Parameters
    p.add_argument("--recluster-linkage", type=str, default="ward",
                   choices=["ward", "complete", "average", "single"],
                   help="Agglomerative clustering linkage")
    
    # Spectral Parameters
    p.add_argument("--recluster-affinity", type=str, default="rbf",
                   choices=["rbf", "nearest_neighbors", "precomputed"],
                   help="Spectral clustering affinity")
    p.add_argument("--recluster-gamma", type=float, default=1.0,
                   help="Spectral clustering gamma parameter")
    
    # GMM Parameters
    p.add_argument("--recluster-covariance-type", type=str, default="full",
                   choices=["full", "tied", "diag", "spherical"],
                   help="GMM covariance type")
    
    # DBSCAN/OPTICS Parameters
    p.add_argument("--recluster-eps", type=float, default=0.5,
                   help="DBSCAN eps parameter")
    p.add_argument("--recluster-min-samples", type=int, default=5,
                   help="DBSCAN/OPTICS min_samples parameter")
    p.add_argument("--recluster-max-eps", type=float, default=float('inf'),
                   help="OPTICS max_eps parameter")
    
    # Birch Parameters
    p.add_argument("--recluster-threshold", type=float, default=0.5,
                   help="Birch threshold parameter")
    p.add_argument("--recluster-branching-factor", type=int, default=50,
                   help="Birch branching_factor parameter")
    
    return p.parse_args()

# Alias for compatibility with cluster_test.py
def parameter_parser():
    return parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")