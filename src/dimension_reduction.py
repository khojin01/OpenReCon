"""
Dimensionality reduction utilities for visualization
"""
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def reduce_dimensions(embeddings, method='tsne', n_components=2, random_state=42):
    """
    Reduce high-dimensional embeddings to 2D for visualization
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        method: 'tsne', 'umap', or 'pca'
        n_components: number of output dimensions (default: 2)
        random_state: random seed for reproducibility
    
    Returns:
        numpy array of shape (n_samples, n_components)
    """
    if embeddings.shape[0] < 4:
        # Too few samples, return as is or pad
        if embeddings.shape[1] >= n_components:
            return embeddings[:, :n_components]
        else:
            padded = np.zeros((embeddings.shape[0], n_components))
            padded[:, :embeddings.shape[1]] = embeddings
            return padded
    
    # First reduce to reasonable dimension with PCA if needed
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50, random_state=random_state)
        embeddings = pca.fit_transform(embeddings)
    
    if method == 'tsne':
        # t-SNE: good for preserving local structure
        perplexity = min(30, max(5, embeddings.shape[0] // 3))
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
            init='pca'
        )
        return tsne.fit_transform(embeddings)
    
    elif method == 'umap' and UMAP_AVAILABLE:
        # UMAP: preserves both local and global structure
        n_neighbors = min(15, max(2, embeddings.shape[0] // 10))
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=random_state
        )
        return reducer.fit_transform(embeddings)
    
    elif method == 'pca' or (method == 'umap' and not UMAP_AVAILABLE):
        # PCA: fast, preserves global structure
        if method == 'umap' and not UMAP_AVAILABLE:
            print("UMAP not available, falling back to PCA")
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(embeddings)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne', 'umap', or 'pca'")


def create_graph_embedding(A_p, A_n, method='spectral', n_components=128):
    """
    Create node embeddings from graph structure
    
    Args:
        A_p: Positive adjacency matrix
        A_n: Negative adjacency matrix
        method: 'spectral' or 'laplacian'
        n_components: embedding dimension
    
    Returns:
        numpy array of shape (n_nodes, n_components)
    """
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD
    
    n_nodes = A_p.shape[0]
    
    # Combine positive and negative edges
    A_combined = A_p - 0.5 * A_n
    
    if hasattr(A_combined, 'toarray'):
        A_combined = A_combined.toarray()
    
    # Use SVD for spectral embedding
    n_comp = min(n_components, n_nodes - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    embeddings = svd.fit_transform(A_combined)
    
    return embeddings
