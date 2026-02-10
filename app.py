import streamlit as st
import torch
import numpy as np
import pickle as pk
import scipy.sparse as ss
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import sys
import os
from io import BytesIO
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
src_path = os.path.join(current_dir, 'src')
signet_path = os.path.join(src_path, 'SigNet')
sys.path.append(signet_path)

from signet.cluster import Cluster
from src.metrics import calculate_acc, compute_modularity, compute_conductance, calculate_unhappy_ratio
from src.initial_clustering import run_initial_clustering
from src.pipeline import run_full_pipeline
from src.visualization_callback import reset_tracker, get_tracker
from src.visualization_components import (
    create_refinement_animation, 
    create_iteration_timeline,
    create_cluster_evolution_chart,
    create_sankey_diagram,
    create_detailed_stats_table
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

st.set_page_config(
    page_title="ReCon: Signed Network Clustering",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일시트 로드
def load_css():
    with open(".streamlit/style.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def create_network_visualization(A_p, A_n, clusters, embeddings=None, max_nodes=500, method='tsne'):
    """네트워크 시각화 생성 (실제 임베딩 기반)"""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA, TruncatedSVD
    
    n_nodes = min(A_p.shape[0], max_nodes)
    
    A_p_dense = A_p[:n_nodes, :n_nodes].toarray() if hasattr(A_p, 'toarray') else A_p[:n_nodes, :n_nodes]
    A_n_dense = A_n[:n_nodes, :n_nodes].toarray() if hasattr(A_n, 'toarray') else A_n[:n_nodes, :n_nodes]
    
    # Use actual embeddings with dimensionality reduction
    if embeddings is not None and embeddings.shape[0] >= n_nodes:
        emb_subset = embeddings[:n_nodes]
        # Reduce to 2D
        if emb_subset.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            emb_subset = pca.fit_transform(emb_subset)
        
        perplexity = min(30, max(5, n_nodes // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000, init='pca')
        coords = tsne.fit_transform(emb_subset)
        pos_x, pos_y = coords[:, 0], coords[:, 1]
    else:
        # Fallback: create embeddings from graph structure
        A_combined = A_p[:n_nodes, :n_nodes] - 0.5 * A_n[:n_nodes, :n_nodes]
        if hasattr(A_combined, 'toarray'):
            A_combined = A_combined.toarray()
        
        svd = TruncatedSVD(n_components=min(50, n_nodes-1), random_state=42)
        emb_subset = svd.fit_transform(A_combined)
        
        perplexity = min(30, max(5, n_nodes // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000, init='pca')
        coords = tsne.fit_transform(emb_subset)
        pos_x, pos_y = coords[:, 0], coords[:, 1]
    
    edge_trace_pos = []
    edge_trace_neg = []
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if A_p_dense[i, j] > 0:
                edge_trace_pos.append(
                    go.Scatter(
                        x=[pos_x[i], pos_x[j], None],
                        y=[pos_y[i], pos_y[j], None],
                        mode='lines',
                        line=dict(width=0.5, color='green'),
                        hoverinfo='none',
                        showlegend=False
                    )
                )
            if A_n_dense[i, j] > 0:
                edge_trace_neg.append(
                    go.Scatter(
                        x=[pos_x[i], pos_x[j], None],
                        y=[pos_y[i], pos_y[j], None],
                        mode='lines',
                        line=dict(width=0.5, color='red', dash='dash'),
                        hoverinfo='none',
                        showlegend=False
                    )
                )
    
    node_trace = go.Scatter(
        x=pos_x,
        y=pos_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=10,
            color=clusters[:n_nodes],
            colorbar=dict(
                thickness=15,
                title=dict(text='Cluster', side='right'),
                xanchor='left'
            ),
            line=dict(width=1, color='white')
        ),
        text=[f'Node {i}<br>Cluster {clusters[i]}' for i in range(n_nodes)]
    )
    
    fig = go.Figure(data=edge_trace_pos + edge_trace_neg + [node_trace],
                    layout=go.Layout(
                        title=f'Network Visualization (showing {n_nodes} nodes)',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500
                    ))
    
    return fig

def create_cluster_distribution_chart(clusters):
    """클러스터 분포 차트 생성"""
    cluster_counts = Counter(clusters)
    df = pd.DataFrame(list(cluster_counts.items()), columns=['Cluster', 'Count'])
    df = df.sort_values('Cluster')
    
    fig = px.bar(df, x='Cluster', y='Count', 
                 title='Cluster Size Distribution',
                 labels={'Count': 'Number of Nodes'},
                 color='Count',
                 color_continuous_scale='Blues')
    
    fig.update_layout(height=400)
    return fig

def create_metrics_comparison_chart(initial_metrics, final_metrics):
    """메트릭 비교 차트 생성"""
    metrics_names = ['ACC', 'ARI', 'NMI', 'Modularity', 'Conductance', 'Unhappy Ratio']
    
    initial_values = [
        initial_metrics.get('acc', 0),
        initial_metrics.get('ari', 0),
        initial_metrics.get('nmi', 0),
        initial_metrics.get('modularity', 0),
        initial_metrics.get('conductance', 0),
        initial_metrics.get('unhappy', 0)
    ]
    
    final_values = [
        final_metrics.get('acc', 0),
        final_metrics.get('ari', 0),
        final_metrics.get('nmi', 0),
        final_metrics.get('modularity', 0),
        final_metrics.get('conductance', 0),
        final_metrics.get('unhappy', 0)
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='Initial', x=metrics_names, y=initial_values, marker_color='lightblue'),
        go.Bar(name='Final', x=metrics_names, y=final_values, marker_color='darkblue')
    ])
    
    fig.update_layout(
        title='Performance Metrics Comparison',
        barmode='group',
        height=400,
        yaxis_title='Score'
    )
    
    return fig

def main():
    st.markdown('<p class="main-header">🔗 ReCon: Signed Network Clustering</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Community Detection on Signed Networks via Refinement and Contrastive Learning</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📁 Data Input")
        data_source = st.radio("Data Source", ["Upload File", "Example Data"])
        
        if data_source == "Upload File":
            uploaded_file = st.file_uploader("Upload data file (.pt or .pk)", type=['pt', 'pk'])
            data_path = None
            if uploaded_file is not None:
                file_extension = uploaded_file.name.split('.')[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    data_path = tmp_file.name
        else:
            example_files = [f for f in os.listdir('data/SSBM') if f.endswith(('.pt', '.pk'))] if os.path.exists('data/SSBM') else []
            if example_files:
                selected_file = st.selectbox("Select Example", example_files)
                data_path = os.path.join('data/SSBM', selected_file)
            else:
                st.warning("No example data found in data/SSBM/")
                data_path = None
        
        st.subheader("🎯 Clustering Parameters")
        K = st.number_input("Number of Clusters (K)", min_value=2, max_value=20, value=5, 
                           help="Number of communities to detect")
        
        initial_method = st.selectbox(
            "Initial Clustering Method",
            ["SPONGE", "SPONGE_sym", "sssnet", "spectral_cluster_adjacency", "dsgc"],
            help="Algorithm for initial clustering"
        )
        
        iterations = st.slider("Pipeline Iterations", min_value=1, max_value=10, value=3,
                              help="Number of refinement iterations")
        
        st.subheader("🔧 Advanced Settings")
        with st.expander("Refinement Parameters"):
            refine_alpha = st.slider("Alpha (Graph Weight)", 0.0, 1.0, 0.8, 0.1)
            refine_beta = st.slider("Beta (Embedding Weight)", 0.0, 1.0, 0.2, 0.1)
            refine_neg_weight = st.slider("Negative Edge Weight", 1.0, 5.0, 3.0, 0.5)
        
        with st.expander("Re-clustering Parameters"):
            recluster_method = st.selectbox(
                "Re-clustering Method",
                ["kmeans", "agglomerative", "spectral", "spectral_cosine"]
            )
        
        device_option = st.selectbox("Device", ["auto", "cpu", "cuda"])
        seed = st.number_input("Random Seed", min_value=0, value=42)
        
        run_button = st.button("🚀 Run Clustering", type="primary", width='stretch')
    
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    if run_button and data_path:
        try:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            if device_option == 'auto':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device(device_option)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📂 Loading data...")
            progress_bar.progress(10)
            
            # Support both .pk and .pt files with auto-detection
            def load_data_file(file_path):
                """Load data file with automatic format detection"""
                # First try based on extension
                if file_path.endswith('.pk'):
                    try:
                        with open(file_path, "rb") as f:
                            return pk.load(f)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load .pk file: {e}")
                
                # For .pt files, try torch.load first, then fallback to pickle
                try:
                    return torch.load(file_path, map_location='cpu', weights_only=False)
                except RuntimeError as e:
                    if "Invalid magic number" in str(e):
                        # Try as pickle without showing warning (common case)
                        try:
                            with open(file_path, "rb") as f:
                                data = pk.load(f)
                                st.info("📂 Loaded as pickle format (common for research datasets)")
                                return data
                        except Exception as pk_e:
                            st.error(f"❌ Failed to load as both .pt and .pk file:")
                            st.code(f"Torch error: {e}")
                            st.code(f"Pickle error: {pk_e}")
                            st.info("💡 Please ensure your file is either:")
                            st.code("1. A valid PyTorch .pt file")
                            st.code("2. A pickle file (rename to .pk)")
                            st.stop()
                    else:
                        raise e
                except Exception as e:
                    raise RuntimeError(f"Failed to load file: {e}")
            
            data = load_data_file(data_path)
            
            A_p, A_n = data['A_p'], data['A_n']
            feat_L = data.get('feat_L')
            labels = data.get('labels') or data.get('y')
            
            if feat_L is None:
                feat_L = ss.eye(A_p.shape[0])
            
            st.info(f"✅ Loaded network: {A_p.shape[0]} nodes, {A_p.nnz} positive edges, {A_n.nnz} negative edges")
            
            status_text.text(f"🎯 Running initial clustering ({initial_method})...")
            progress_bar.progress(30)
            
            class Args:
                pass
            
            args = Args()
            args.initial_method = initial_method
            args.iterations = iterations
            args.refine_alpha = refine_alpha
            args.refine_beta = refine_beta
            args.refine_neg_weight = refine_neg_weight
            args.recluster_method = recluster_method
            args.seed = seed
            args.device = str(device)
            args.K = K
            args.normalisation = "sym"
            args.solver = "BM_proj_grad"
            args.max_samples_per_edge = 100
            args.lambda_purge = 0.1
            args.s_min_import = 0.3
            args.p_min_import = 0.8
            args.top_m_ratio = 0.5
            args.max_cluster_membership = 10
            args.max_overlap_ratio = 0.5
            args.min_cluster_size = 5
            args.n_jobs = -1
            args.refinement_method = "soft_label_propagation"
            args.refine_temp = 1.0
            args.refine_temp_decay = 0.8
            args.refine_stochastic = False
            args.refine_debug = False
            args.refine_unbalanced_threshold = 0.5
            args.recluster_n_init = 10
            args.recluster_max_iter = 300
            args.recluster_linkage = "ward"
            args.recluster_affinity = "rbf"
            args.recluster_gamma = 1.0
            args.dsgc_epochs = 700
            args.dsgc_hidden = 32
            args.dsgc_dropout = 0.5
            args.dsgc_hop = 2
            args.dsgc_m_p = 3
            args.dsgc_m_n = 2
            args.dsgc_tau = 0.0
            args.dsgc_lr = 0.01
            args.dsgc_weight_decay = 5e-4
            args.dsgc_feature_type = "A_reg"
            args.dsgc_pbnc_lambda = 0.03
            args.dsgc_delta_p = 1
            args.dsgc_delta_n = 1
            
            cluster_model = Cluster((A_p, A_n))
            initial_clusters = run_initial_clustering(
                initial_method, A_p, A_n, feat_L, K, device, args, cluster_model
            )
            
            progress_bar.progress(50)
            
            status_text.text("🔄 Running refinement pipeline...")
            
            # Initialize tracker for visualization
            tracker = reset_tracker()
            
            final_clusters, final_embeddings = run_full_pipeline(args, A_p, A_n, feat_L, initial_clusters, device, tracker=tracker)
            
            progress_bar.progress(80)
            
            status_text.text("📊 Computing metrics...")
            
            initial_metrics = {
                'modularity': compute_modularity(initial_clusters, A_p, A_n),
                'conductance': compute_conductance(initial_clusters, A_p, A_n),
                'unhappy': calculate_unhappy_ratio(A_p, A_n, initial_clusters)
            }
            
            final_metrics = {
                'modularity': compute_modularity(final_clusters, A_p, A_n),
                'conductance': compute_conductance(final_clusters, A_p, A_n),
                'unhappy': calculate_unhappy_ratio(A_p, A_n, final_clusters)
            }
            
            if labels is not None:
                initial_metrics['acc'] = calculate_acc(labels, initial_clusters)
                initial_metrics['ari'] = adjusted_rand_score(labels, initial_clusters)
                initial_metrics['nmi'] = normalized_mutual_info_score(labels, initial_clusters)
                
                final_metrics['acc'] = calculate_acc(labels, final_clusters)
                final_metrics['ari'] = adjusted_rand_score(labels, final_clusters)
                final_metrics['nmi'] = normalized_mutual_info_score(labels, final_clusters)
            
            progress_bar.progress(100)
            status_text.text("✅ Clustering complete!")
            
            st.session_state.results = {
                'A_p': A_p,
                'A_n': A_n,
                'initial_clusters': initial_clusters,
                'final_clusters': final_clusters,
                'initial_metrics': initial_metrics,
                'final_metrics': final_metrics,
                'labels': labels,
                'embeddings': final_embeddings,
                'refinement_history': tracker.get_history(),
                'refinement_summary': tracker.get_summary()
            }
            
            st.success("🎉 Clustering completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    if st.session_state.results:
        results = st.session_state.results
        
        st.header("📊 Results")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Metrics", "🎨 Visualizations", "🔄 Refinement Process", "📋 Cluster Details", "💾 Export"])
        
        with tab1:
            st.subheader("Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Initial Clustering")
                metrics_df_initial = pd.DataFrame([results['initial_metrics']]).T
                metrics_df_initial.columns = ['Value']
                st.dataframe(metrics_df_initial.round(4), use_container_width=True)
            
            with col2:
                st.markdown("### Final Clustering (After ReCon)")
                metrics_df_final = pd.DataFrame([results['final_metrics']]).T
                metrics_df_final.columns = ['Value']
                st.dataframe(metrics_df_final.round(4), use_container_width=True)
            
            st.plotly_chart(
                create_metrics_comparison_chart(results['initial_metrics'], results['final_metrics']),
                use_container_width=True
            )
            
            if results['labels'] is not None:
                improvement = {
                    'ACC': results['final_metrics']['acc'] - results['initial_metrics']['acc'],
                    'ARI': results['final_metrics']['ari'] - results['initial_metrics']['ari'],
                    'NMI': results['final_metrics']['nmi'] - results['initial_metrics']['nmi']
                }
                
                st.markdown("### 📈 Improvement")
                cols = st.columns(3)
                for idx, (metric, value) in enumerate(improvement.items()):
                    with cols[idx]:
                        delta_color = "normal" if value >= 0 else "inverse"
                        st.metric(metric, f"{results['final_metrics'][metric.lower()]:.4f}", 
                                 f"{value:+.4f}", delta_color=delta_color)
        
        with tab2:
            st.subheader("Network Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Initial Clustering")
                st.plotly_chart(
                    create_cluster_distribution_chart(results['initial_clusters']),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### Final Clustering")
                st.plotly_chart(
                    create_cluster_distribution_chart(results['final_clusters']),
                    use_container_width=True
                )

            history = results.get('refinement_history', [])
            if history:
                st.markdown("### 🎬 Animated Refinement Process")
                st.info("**Symbols:** ⭐ Imported nodes | ❌ Purged nodes | 💎 Refined nodes | ⚫ All other nodes (color shows current cluster) | **Each cluster has unique colors**")

                with st.spinner("Creating animation with t-SNE dimensionality reduction..."):
                    animation_fig = create_refinement_animation(
                        history,
                        results['A_p'],
                        results['A_n'],
                        embeddings=results.get('embeddings'),
                        max_nodes=500,
                        method='tsne',
                        final_clusters=results.get('final_clusters'),
                        final_embeddings=results.get('embeddings')
                    )
                    if animation_fig:
                        st.plotly_chart(animation_fig, use_container_width=True)
                        st.caption("📍 Positions and colors update each iteration based on new embeddings and cluster assignments. Use Play/Pause and slider to navigate.")
            else:
                st.info("Animated refinement requires refinement history. Run the pipeline with refinement enabled.")
        
        with tab3:
            st.subheader("🔄 Refinement Process Visualization")
            
            history = results.get('refinement_history', [])
            summary = results.get('refinement_summary', {})
            
            if not history:
                st.warning("No refinement history available. The tracker may not have been enabled during execution.")
            else:
                # Detailed stats table (moved to top)
                st.markdown("### 📋 Detailed Statistics")
                stats_df = create_detailed_stats_table(history)
                if stats_df is not None:
                    st.dataframe(stats_df, width='stretch')

                st.markdown("---")

                # Summary statistics
                st.markdown("### 📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Iterations", summary.get('total_iterations', 0))
                with col2:
                    st.metric("Total Purged", summary.get('total_purged', 0), 
                             help="Nodes removed from clusters due to high unhappiness")
                with col3:
                    st.metric("Total Imported", summary.get('total_imported', 0),
                             help="Nodes added to clusters via triangle participation")
                with col4:
                    st.metric("Total Refined", summary.get('total_refined', 0),
                             help="Nodes that changed clusters during label propagation")
                
                st.markdown("---")
                
                # Timeline chart
                st.markdown("### 📈 Changes per Iteration")
                timeline_fig = create_iteration_timeline(history)
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)
                    st.caption("**Legend:** 🔴 Purged = removed from clusters | 🟢 Imported = added to clusters | 🔵 Refined = moved between clusters")
                
                # Cluster evolution
                st.markdown("### 📊 Cluster Size Evolution")
                evolution_fig = create_cluster_evolution_chart(history)
                if evolution_fig:
                    st.plotly_chart(evolution_fig, use_container_width=True)
                    st.caption("Track how cluster sizes change across iterations")
                
                # Sankey diagram
                st.markdown("### 🌊 Node Flow Between Clusters")
                sankey_fig = create_sankey_diagram(history, max_iterations=min(3, len(history)))
                if sankey_fig:
                    st.plotly_chart(sankey_fig, use_container_width=True)
                    st.caption("Visualizes how nodes move between clusters across iterations")
                else:
                    st.info("Sankey diagram requires at least 2 iterations")
                
                # Download refinement history
                st.markdown("### 💾 Export Refinement Data")
                if stats_df is not None:
                    refinement_csv = stats_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Refinement Statistics (CSV)",
                        data=refinement_csv,
                        file_name="refinement_statistics.csv",
                        mime="text/csv"
                    )
        
        with tab4:
            st.subheader("Cluster Distribution Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Initial Clustering")
                initial_counts = Counter(results['initial_clusters'])
                initial_df = pd.DataFrame([
                    {'Cluster': k, 'Size': v, 'Percentage': f"{v/len(results['initial_clusters'])*100:.2f}%"}
                    for k, v in sorted(initial_counts.items())
                ])
                st.dataframe(initial_df, width='stretch')
            
            with col2:
                st.markdown("#### Final Clustering")
                final_counts = Counter(results['final_clusters'])
                final_df = pd.DataFrame([
                    {'Cluster': k, 'Size': v, 'Percentage': f"{v/len(results['final_clusters'])*100:.2f}%"}
                    for k, v in sorted(final_counts.items())
                ])
                st.dataframe(final_df, width='stretch')
        
        with tab5:
            st.subheader("Export Results")
            
            st.markdown("#### Download Clustering Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                initial_csv = pd.DataFrame({
                    'node_id': range(len(results['initial_clusters'])),
                    'cluster': results['initial_clusters']
                }).to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Initial Clusters (CSV)",
                    data=initial_csv,
                    file_name="initial_clusters.csv",
                    mime="text/csv"
                )
            
            with col2:
                final_csv = pd.DataFrame({
                    'node_id': range(len(results['final_clusters'])),
                    'cluster': results['final_clusters']
                }).to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Final Clusters (CSV)",
                    data=final_csv,
                    file_name="final_clusters.csv",
                    mime="text/csv"
                )
            
            metrics_csv = pd.DataFrame({
                'Metric': list(results['final_metrics'].keys()),
                'Initial': [results['initial_metrics'].get(k, 0) for k in results['final_metrics'].keys()],
                'Final': list(results['final_metrics'].values())
            }).to_csv(index=False)
            
            st.download_button(
                label="📥 Download Metrics Comparison (CSV)",
                data=metrics_csv,
                file_name="metrics_comparison.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👈 Configure parameters in the sidebar and click 'Run Clustering' to start")
        
        st.markdown("""
        ### 📖 About ReCon
        
        ReCon is a framework for improving community detection accuracy on signed networks through:
        
        - **Community Refinement**: Iterative refinement using soft label propagation
        - **Contrastive Learning**: Learning robust node embeddings
        - **Multiple Initial Methods**: Support for various clustering algorithms (SPONGE, SSSNET, DSGC, etc.)
        
        ### 🚀 Quick Start
        
        1. Upload your signed network data (.pt or .pk file) or select an example
        2. Configure clustering parameters (K, initial method, iterations)
        3. Adjust advanced settings if needed
        4. Click "Run Clustering" to start the analysis
        5. View results in the tabs above
        
        ### 📊 Supported Metrics
        
        - **ACC**: Clustering Accuracy
        - **ARI**: Adjusted Rand Index
        - **NMI**: Normalized Mutual Information
        - **Modularity**: Network modularity score
        - **Conductance**: Average cluster conductance
        - **Unhappy Ratio**: Ratio of nodes with more negative than positive edges within cluster
        
        ### 📁 Supported Data Formats
        
        - **.pt files**: PyTorch tensor format (torch.load)
        - **.pk files**: Pickle format (pickle.load)
        
        Both formats should contain a dictionary with:
        - `A_p`: Positive adjacency matrix
        - `A_n`: Negative adjacency matrix  
        - `feat_L` (optional): Node features
        - `labels` or `y` (optional): Ground truth labels
        """)

if __name__ == "__main__":
    main()
