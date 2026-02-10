import torch
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

class SignedKMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, adj_pos=None, adj_neg=None):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centers = X[indices]

        for it in range(self.max_iter):
            dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)

            if adj_pos is not None:
                for i in range(n_samples):
                    for j in range(n_samples):
                        if adj_pos[i, j]:
                            dists[i] -= 0.5
            if adj_neg is not None:
                for i in range(n_samples):
                    for j in range(n_samples):
                        if adj_neg[i, j]:
                            dists[i] += 0.5

            labels = np.argmin(dists, axis=1)

            new_centers = np.array([X[labels==k].mean(axis=0) if np.any(labels==k) else centers[k] 
                                    for k in range(self.n_clusters)])
            
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            centers = new_centers

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

def recluster_embedding(embedding, n_clusters, method='kmeans', **kwargs):
    print(f"\n>> Re-clustering embedding of shape {embedding.shape} into {n_clusters} clusters using {method}...")
    
    if isinstance(embedding, torch.Tensor):
        embedding_np = embedding.cpu().detach().numpy()
    else:
        embedding_np = embedding

    if method == 'kmeans':
        kmeans_kwargs = {'random_state': 42, 'n_init': 10}
        valid_kmeans_params = ['n_init', 'max_iter', 'random_state']
        for key, value in kwargs.items():
            if key in valid_kmeans_params:
                kmeans_kwargs[key] = value
        kmeans = KMeans(n_clusters=n_clusters, **kmeans_kwargs)
        cluster_labels = kmeans.fit_predict(embedding_np)
    
    elif method == 'signed_kmeans':
        adj_pos = kwargs.get('adj_pos', None)
        adj_neg = kwargs.get('adj_neg', None)
        signed_kmeans = SignedKMeans(n_clusters=n_clusters,
                                        max_iter=kwargs.get('max_iter', 100),
                                        tol=kwargs.get('tol', 1e-4),
                                        random_state=kwargs.get('random_state', 42))
        signed_kmeans.fit(embedding_np, adj_pos=adj_pos, adj_neg=adj_neg)
        cluster_labels = signed_kmeans.labels_


    elif method == 'spectral':
        adj_pos = kwargs.get('adj_pos', None)
        adj_neg = kwargs.get('adj_neg', None)

        if adj_pos is not None and adj_neg is not None:
            adj = adj_pos - adj_neg
            from scipy.sparse import csgraph
            L = csgraph.laplacian(adj, normed=True)
            L = np.nan_to_num(L, nan=0.0, posinf=1e6, neginf=-1e6)
            affinity = np.exp(-rbf_kernel(L))
            affinity = np.nan_to_num(affinity, nan=0.0)
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42,
                n_init=10
            )
            cluster_labels = spectral.fit_predict(affinity)
        else:
            rbf_sim = rbf_kernel(embedding_np, gamma=kwargs.get('gamma', 1.0))
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42,
                n_init=10
            )
            cluster_labels = spectral.fit_predict(rbf_sim)



    elif method == 'spectral_cosine':
        if np.any(np.isnan(embedding_np)):
            print(f"Warning: Embedding contains NaN values, falling back to K-Means")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
            cluster_labels = kmeans.fit_predict(embedding_np)
        else:
            cosine_sim = cosine_similarity(embedding_np)
            cosine_sim = np.maximum(cosine_sim, 0)
            spectral_kwargs = {'random_state': 42, 'n_init': 10}
            spectral_kwargs.update(kwargs)
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', **spectral_kwargs)
            cluster_labels = spectral.fit_predict(cosine_sim)
    
    elif method == 'spectral_rbf':
        if np.any(np.isnan(embedding_np)):
            print(f"Warning: Embedding contains NaN values, falling back to K-Means")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
            cluster_labels = kmeans.fit_predict(embedding_np)
        else:
            rbf_sim = rbf_kernel(embedding_np, gamma=kwargs.get('gamma', 1.0))
            spectral_kwargs = {'random_state': 42, 'n_init': 10}
            spectral_kwargs.update(kwargs)
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', **spectral_kwargs)
            cluster_labels = spectral.fit_predict(rbf_sim)
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    print(f">> Re-clustering complete using {method}.")
    return cluster_labels
