import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import hex_to_rgb
import pandas as pd
from typing import Dict, List, Optional
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

def _reduce_dimensions(embeddings, method='tsne'):
    """Reduce embeddings to 2D using t-SNE or UMAP"""
    from sklearn.decomposition import PCA
    
    if embeddings.shape[0] < 4:
        # Too few samples
        if embeddings.shape[1] >= 2:
            return embeddings[:, 0], embeddings[:, 1]
        else:
            return np.zeros(embeddings.shape[0]), np.zeros(embeddings.shape[0])
    
    # First reduce to 50D with PCA if needed
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        embeddings = pca.fit_transform(embeddings)
    
    if method == 'umap' and UMAP_AVAILABLE:
        n_neighbors = min(15, max(2, embeddings.shape[0] // 10))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(embeddings)
        return coords[:, 0], coords[:, 1]
    else:
        # Use t-SNE (default)
        perplexity = min(30, max(5, embeddings.shape[0] // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000, init='pca')
        coords = tsne.fit_transform(embeddings)
        return coords[:, 0], coords[:, 1]

def create_refinement_animation(history: List[Dict], A_p, A_n, embeddings=None, max_nodes=500, method='tsne', final_clusters=None, final_embeddings=None):
    """Create animated visualization of refinement process using actual embeddings
    
    Args:
        history: Refinement history from tracker
        A_p: Positive adjacency matrix
        A_n: Negative adjacency matrix
        embeddings: Node embeddings (if available)
        max_nodes: Maximum nodes to visualize
        method: Dimensionality reduction method ('tsne' or 'umap')
    """
    
    if not history:
        return None
    
    n_nodes = min(A_p.shape[0], max_nodes)
    
    # Get initial clusters (before any refinement) - use the first iteration's clusters as baseline
    initial_clusters = history[0]['clusters'][:n_nodes] if 'clusters' in history[0] else np.zeros(n_nodes)
    
    frames = []
    
    # Add frame for "Before Refinement" (iteration -1)
    # Use initial embeddings or create fallback
    initial_embeddings = history[0].get('embeddings')
    if initial_embeddings is not None and initial_embeddings.shape[0] >= n_nodes:
        emb_subset = initial_embeddings[:n_nodes]
        pos_x, pos_y = _reduce_dimensions(emb_subset, method=method)
    else:
        # Fallback: create embeddings from graph structure
        from scipy.sparse import csr_matrix
        A_combined = A_p[:n_nodes, :n_nodes] - 0.5 * A_n[:n_nodes, :n_nodes]
        if hasattr(A_combined, 'toarray'):
            A_combined = A_combined.toarray()
        
        # Use spectral embedding as fallback
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=min(50, n_nodes-1), random_state=42)
        emb_subset = svd.fit_transform(A_combined)
        pos_x, pos_y = _reduce_dimensions(emb_subset, method=method)
    
    pre_refine_frame = go.Frame(
        data=[go.Scatter(
            x=pos_x,
            y=pos_y,
            mode='markers',
            marker=dict(
                size=np.ones(n_nodes) * 10,
                color=initial_clusters,
                colorscale='Viridis',
                symbol=['circle'] * n_nodes,
                line=dict(width=1, color='white')
            ),
            text=[f'Node {i}<br>Cluster {int(initial_clusters[i])}<br>INITIAL STATE' for i in range(n_nodes)],
            hoverinfo='text'
        )],
        name="Before Refinement",
        layout=go.Layout(
            title_text="Before Refinement - Initial State"
        )
    )
    frames.append(pre_refine_frame)
    
    # Track changes from initial state
    changed_nodes = set()
    
    for idx, iteration_data in enumerate(history):
        clusters = iteration_data.get('clusters_after_refinement', 
                                      iteration_data.get('clusters_after_import',
                                      iteration_data.get('clusters', initial_clusters)))[:n_nodes]
        
        purged = iteration_data.get('purged_nodes', set())
        imported = iteration_data.get('imported_nodes', {})
        refined = iteration_data.get('refined_nodes', {})
        
        # Find nodes that changed from initial state
        current_changed = set()
        for i in range(n_nodes):
            if clusters[i] != initial_clusters[i]:
                current_changed.add(i)
        
        # Use embeddings from this iteration if available
        iteration_embeddings = iteration_data.get('embeddings')
        if iteration_embeddings is not None and iteration_embeddings.shape[0] >= n_nodes:
            emb_subset = iteration_embeddings[:n_nodes]
            iter_pos_x, iter_pos_y = _reduce_dimensions(emb_subset, method=method)
        else:
            # Use initial positions as fallback
            iter_pos_x, iter_pos_y = pos_x, pos_y
        
        # Node colors based on current clusters (each cluster gets unique color)
        node_colors = clusters.copy()
        
        # Node sizes and symbols based on status
        node_sizes = np.ones(n_nodes) * 10
        node_symbols = ['circle'] * n_nodes
        
        for i in range(n_nodes):
            if i in purged:
                node_sizes[i] = 15
                node_symbols[i] = 'x'
                current_changed.add(i)
            elif i in imported:
                node_sizes[i] = 15
                node_symbols[i] = 'star'
                current_changed.add(i)
            elif i in refined:
                node_sizes[i] = 12
                node_symbols[i] = 'diamond'
                current_changed.add(i)
            # Changed nodes get normal circle symbol (color shows new cluster)
        
        frame = go.Frame(
            data=[go.Scatter(
                x=iter_pos_x,
                y=iter_pos_y,
                mode='markers',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='Viridis',
                    symbol=node_symbols,
                    line=dict(width=1, color='white')
                ),
                text=[f'Node {i}<br>Initial Cluster {int(initial_clusters[i])}<br>Current Cluster {int(clusters[i])}<br>' + 
                      ('PURGED' if i in purged else 
                       'IMPORTED' if i in imported else
                       'REFINED' if i in refined else 'STABLE')
                      for i in range(n_nodes)],
                hoverinfo='text'
            )],
            name=f"Iteration {iteration_data.get('iteration', idx)}",
            layout=go.Layout(
                title_text=f"Iteration {iteration_data.get('iteration', idx)} - Stage: {iteration_data.get('stage', 'unknown')}"
            )
        )
        frames.append(frame)

    # Final frame after recluster (if provided)
    if final_clusters is not None:
        final_clusters = final_clusters[:n_nodes]
        if final_embeddings is not None and final_embeddings.shape[0] >= n_nodes:
            final_emb_subset = final_embeddings[:n_nodes]
            final_pos_x, final_pos_y = _reduce_dimensions(final_emb_subset, method=method)
        else:
            final_pos_x, final_pos_y = pos_x, pos_y

        final_frame = go.Frame(
            data=[go.Scatter(
                x=final_pos_x,
                y=final_pos_y,
                mode='markers',
                marker=dict(
                    size=np.ones(n_nodes) * 10,
                    color=final_clusters,
                    colorscale='Viridis',
                    symbol=['circle'] * n_nodes,
                    line=dict(width=1, color='white')
                ),
                text=[f'Node {i}<br>Initial Cluster {int(initial_clusters[i])}<br>Final Cluster {int(final_clusters[i])}<br>FINAL RESULT'
                      for i in range(n_nodes)],
                hoverinfo='text'
            )],
            name="Final (After Recluster)",
            layout=go.Layout(
                title_text="Final (After Recluster) - Final Embeddings"
            )
        )
        frames.append(final_frame)
    
    # Initial figure - show "Before Refinement" state with initial positions
    fig = go.Figure(
        data=[go.Scatter(
            x=pos_x,
            y=pos_y,
            mode='markers',
            marker=dict(
                size=10,
                color=initial_clusters,
                colorscale='Viridis',
                line=dict(width=1, color='white')
            ),
            text=[f'Node {i}<br>Initial Cluster {int(initial_clusters[i])}<br>INITIAL STATE' for i in range(n_nodes)],
            hoverinfo='text'
        )],
        frames=frames
    )
    
    fig.update_layout(
        title="Refinement Process Animation",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 1000, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 300}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 0
        }],
        sliders=[{
            'active': 0,
            'steps': [
                {
                    'args': [[f.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': f.name,
                    'method': 'animate'
                }
                for f in frames
            ],
            'x': 0.1,
            'len': 0.9,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top'
        }]
    )
    
    return fig

def create_iteration_timeline(history: List[Dict]):
    """Create timeline showing purge/import/refine statistics per iteration"""
    
    if not history:
        return None
    
    iterations = []
    purged_counts = []
    imported_counts = []
    refined_counts = []
    
    for it_data in history:
        iterations.append(it_data.get('iteration', 0))
        purged_counts.append(len(it_data.get('purged_nodes', set())))
        imported_counts.append(len(it_data.get('imported_nodes', {})))
        refined_counts.append(len(it_data.get('refined_nodes', {})))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Purged Nodes',
        x=iterations,
        y=purged_counts,
        marker_color='red',
        text=purged_counts,
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='Imported Nodes',
        x=iterations,
        y=imported_counts,
        marker_color='green',
        text=imported_counts,
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='Refined Nodes',
        x=iterations,
        y=refined_counts,
        marker_color='blue',
        text=refined_counts,
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Node Changes per Iteration',
        xaxis_title='Iteration',
        yaxis_title='Number of Nodes',
        barmode='group',
        height=400,
        legend=dict(x=0.7, y=1)
    )
    
    return fig

def create_cluster_evolution_chart(history: List[Dict]):
    """Create chart showing cluster size evolution over iterations"""
    
    if not history:
        return None
    
    # Collect cluster sizes over iterations
    iteration_nums = []
    cluster_data = {}
    
    for it_data in history:
        it_num = it_data.get('iteration', 0)
        iteration_nums.append(it_num)
        
        clusters = it_data.get('clusters_after_refinement', 
                              it_data.get('clusters_after_import',
                              it_data.get('clusters')))
        
        if clusters is not None:
            unique, counts = np.unique(clusters[clusters != -1], return_counts=True)
            for cluster_id, count in zip(unique, counts):
                if cluster_id not in cluster_data:
                    cluster_data[cluster_id] = []
                cluster_data[cluster_id].append((it_num, count))
    
    fig = go.Figure()
    
    for cluster_id, data_points in cluster_data.items():
        data_points.sort(key=lambda x: x[0])
        iters, sizes = zip(*data_points)
        
        fig.add_trace(go.Scatter(
            x=iters,
            y=sizes,
            mode='lines+markers',
            name=f'Cluster {cluster_id}',
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Cluster Size Evolution',
        xaxis_title='Iteration',
        yaxis_title='Cluster Size',
        height=400,
        legend=dict(x=1.05, y=1, xanchor='left')
    )
    
    return fig

def create_sankey_diagram(history: List[Dict], max_iterations=3):
    """Create Sankey diagram showing node flow between clusters"""
    
    if len(history) < 2:
        return None
    
    # Limit to first few iterations for clarity
    history_subset = history[:max_iterations]
    
    # Build node labels and links
    labels = []
    sources = []
    targets = []
    values = []
    colors = []
    node_colors = []
    
    color_map = px.colors.qualitative.Plotly

    def _to_rgba(color: str, alpha: float = 0.35) -> str:
        if color.startswith("rgb"):
            return color.replace("rgb", "rgba").replace(")", f", {alpha})")
        r, g, b = hex_to_rgb(color)
        return f"rgba({r}, {g}, {b}, {alpha})"
    
    # Create labels for each cluster at each iteration
    label_map = {}
    label_idx = 0
    
    for it_idx, it_data in enumerate(history_subset):
        clusters = it_data.get('clusters_after_refinement', 
                              it_data.get('clusters_after_import',
                              it_data.get('clusters')))
        
        if clusters is not None:
            unique_clusters = np.unique(clusters[clusters != -1])
            for cluster_id in unique_clusters:
                label = f"Iter {it_data.get('iteration', it_idx)}<br>Cluster {cluster_id}"
                labels.append(label)
                node_colors.append(color_map[cluster_id % len(color_map)])
                label_map[(it_idx, cluster_id)] = label_idx
                label_idx += 1
    
    # Create links between consecutive iterations
    for i in range(len(history_subset) - 1):
        clusters_before = history_subset[i].get('clusters_after_refinement',
                                                history_subset[i].get('clusters'))
        clusters_after = history_subset[i+1].get('clusters_after_refinement',
                                                 history_subset[i+1].get('clusters'))
        
        if clusters_before is not None and clusters_after is not None:
            # Count transitions
            transitions = {}
            for node_idx in range(len(clusters_before)):
                if clusters_before[node_idx] != -1 and clusters_after[node_idx] != -1:
                    key = (int(clusters_before[node_idx]), int(clusters_after[node_idx]))
                    transitions[key] = transitions.get(key, 0) + 1
            
            # Add links
            for (from_cluster, to_cluster), count in transitions.items():
                if (i, from_cluster) in label_map and (i+1, to_cluster) in label_map:
                    sources.append(label_map[(i, from_cluster)])
                    targets.append(label_map[(i+1, to_cluster)])
                    values.append(count)
                    colors.append(_to_rgba(color_map[from_cluster % len(color_map)], 0.45))
    
    if not sources:
        return None
    
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=25,
            thickness=28,
            line=dict(color='rgba(0,0,0,0.6)', width=1),
            label=labels,
            color=node_colors,
            hovertemplate="Cluster: %{label}<extra></extra>"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
            hovertemplate="From: %{source.label}<br>To: %{target.label}<br>Nodes: %{value}<extra></extra>"
        )
    )])
    
    fig.update_layout(
        title="Node Flow Between Clusters",
        height=700,
        font_size=14,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_detailed_stats_table(history: List[Dict]):
    """Create detailed statistics table for each iteration"""
    
    if not history:
        return None
    
    data = []
    
    for it_data in history:
        iteration = it_data.get('iteration', 0)
        stage = it_data.get('stage', 'unknown')
        
        clusters = it_data.get('clusters_after_refinement',
                              it_data.get('clusters_after_import',
                              it_data.get('clusters')))
        
        n_clusters = len(np.unique(clusters[clusters != -1])) if clusters is not None else 0
        n_purged = len(it_data.get('purged_nodes', set()))
        n_imported = len(it_data.get('imported_nodes', {}))
        n_refined = len(it_data.get('refined_nodes', {}))
        
        data.append({
            'Iteration': iteration,
            'Stage': stage,
            'Clusters': n_clusters,
            'Purged': n_purged,
            'Imported': n_imported,
            'Refined': n_refined,
            'Total Changes': n_purged + n_imported + n_refined
        })
    
    df = pd.DataFrame(data)
    return df
