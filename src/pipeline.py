import numpy as np
from collections import Counter

def run_full_pipeline(args, A_p, A_n, feat_L, initial_clusters, device, tracker=None):
    print(f"\n[Running Pipeline] iterations={args.iterations}")
    
    k = len(np.unique(initial_clusters))
    current_clusters = initial_clusters.copy()
    final_embeddings = None  # Store final embeddings for visualization
    
    if tracker is not None:
        tracker.reset()
        tracker.record_initial_state(current_clusters)
    
    A_p_lil = A_p.tolil().copy()
    A_n_lil = A_n.tolil().copy()
    
    if args.refinement_method == "soft_label_propagation":
        from src.structural_refine import DenseSignedIterativeRefinement
        refinement_algorithm = DenseSignedIterativeRefinement()
    elif args.refinement_method == "edge_flipping":
        from cluster_refinement.edge_flipping_soft_label_propagation import EdgeFlippingSoftLabelPropagation
        refinement_algorithm = EdgeFlippingSoftLabelPropagation()
    else:
        raise ValueError(f"Unknown refinement method: {args.refinement_method}")
    
    from src.boundary_refine import CommunityConstructor
    cluster_constructor = CommunityConstructor()
    
    print("  Initial refinement...")
    refine_kwargs = {}
    if args.refinement_method == "edge_flipping":
        refine_kwargs['unbalanced_threshold'] = args.refine_unbalanced_threshold
    
    current_clusters = refinement_algorithm.refine(
        clusters=current_clusters,
        A_p=A_p_lil,
        A_n=A_n_lil,
        T=5,
        alpha=args.refine_alpha,
        beta=args.refine_beta,
        neg_weight=args.refine_neg_weight,
        tracker=tracker,
        **refine_kwargs
    )
    
    for i in range(args.iterations):
        print(f"  Iteration {i+1}/{args.iterations}...")
        
        if tracker is not None:
            tracker.start_iteration(i + 1)
        
        cluster_index, num_nodes_out, num_clusters_out, new_clusters, cluster_sets = cluster_constructor.construct(
            clusters=current_clusters,
            A_p=A_p,
            A_n=A_n,
            tracker=tracker,
            lambda_purge=args.lambda_purge,
            s_min_import=args.s_min_import,
            p_min_import=args.p_min_import
        )
        
        from src.embedding import run_cl_embedding
        node_embeddings = run_cl_embedding(
            features=feat_L,
            cluster_index=cluster_index,
            num_nodes=num_nodes_out,
            num_clusters=num_clusters_out,
            device=device
        )
        
        # Record embeddings in tracker
        if tracker is not None:
            tracker.record_embeddings(node_embeddings)
        
        from src.recluster import recluster_embedding
        current_clusters = recluster_embedding(embedding=node_embeddings, n_clusters=k, random_state=args.seed)
        
        # Store embeddings from last iteration for visualization
        final_embeddings = node_embeddings
        
        cluster_counts = Counter(current_clusters)
        sorted_counts = dict(sorted(cluster_counts.items()))
        
        
        if i < args.iterations - 1:
            current_clusters = refinement_algorithm.refine(
                clusters=current_clusters,
                A_p=A_p_lil,
                A_n=A_n_lil,
                T=5,
                alpha=args.refine_alpha,
                beta=args.refine_beta,
                neg_weight=args.refine_neg_weight,
                tracker=tracker,
                **refine_kwargs
            )
            
            cluster_counts = Counter(current_clusters)
            sorted_counts = dict(sorted(cluster_counts.items()))
        else:
            print(f"    Skipping refinement in final iteration (iteration {i+1}/{args.iterations})")
        
        if tracker is not None:
            tracker.finalize_iteration()
    
    return current_clusters, final_embeddings
