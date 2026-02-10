import numpy as np
from scipy.sparse import issparse, csr_matrix, csc_matrix
from sklearn.preprocessing import normalize

class DenseSignedIterativeRefinement:
    def refine(
        self,
        clusters,
        A_p,
        A_n,
        embeddings=None,
        T: int = 5,
        debug=False,
        alpha=None,
        beta=None,
        neg_weight=None,
        temp_init=1.0,
        temp_decay=0.8,
        stochastic=False,
        tracker=None,
        **kwargs
    ):
        alpha = alpha if alpha is not None else kwargs.get('refine_alpha')
        beta = beta if beta is not None else kwargs.get('refine_beta')
        neg_weight = neg_weight if neg_weight is not None else kwargs.get('refine_neg_weight')

        temp_init = temp_init or kwargs.get('refine_temp', 1.0)
        temp_decay = temp_decay or kwargs.get('refine_temp_decay', 0.8)
        stochastic = stochastic or kwargs.get('refine_stochastic', False)

        labels = np.array(clusters, dtype=int).copy()
        labels_before = labels.copy()
        num_nodes = len(labels)
        k = int(np.max(labels)) + 1
        
        refined_nodes = {}

        total = alpha + beta
        if total > 0:
            alpha /= total
            beta /= total
        else:
            alpha = 0.5
            beta = 0.5

        def to_csr(x):
            if x is None:
                return None
            if isinstance(x, csr_matrix):
                return x
            if isinstance(x, csc_matrix):
                return x.tocsr()
            if issparse(x):
                return x.tocsr()
            return x

        A_p = to_csr(A_p)
        A_n = to_csr(A_n)

        if embeddings is None:
            A_signed = (A_p if A_p is not None else np.zeros((num_nodes, num_nodes))) -                       (A_n if A_n is not None else np.zeros((num_nodes, num_nodes)))
            embeddings = normalize(A_signed, axis=1) if not issparse(A_signed) else normalize(A_signed.toarray(), axis=1)

        def neighbors_weights(A):
            if A is None:
                return [np.array([], dtype=int) for _ in range(num_nodes)], [None]*num_nodes
            if issparse(A):
                neigh = [A.indices[A.indptr[i]:A.indptr[i+1]] for i in range(num_nodes)]
                weights = [A.data[A.indptr[i]:A.indptr[i+1]] for i in range(num_nodes)]
            else:
                neigh = [np.where(A[i] > 0)[0] for i in range(num_nodes)]
                weights = [A[i, neigh[i]] for i in range(num_nodes)]
            return neigh, weights

        pos_neighbors, pos_weights = neighbors_weights(A_p)
        neg_neighbors, neg_weights = neighbors_weights(A_n)

        rng = np.random.default_rng()

        for it in range(T):
            changes = 0
            order = np.arange(num_nodes)
            rng.shuffle(order)

            cluster_centers = np.zeros((k, embeddings.shape[1]), dtype=np.float32)
            for c in range(k):
                idx = np.where(labels == c)[0]
                if idx.size > 0:
                    cluster_centers[c] = embeddings[idx].mean(axis=0)
            cluster_centers = normalize(cluster_centers, axis=1)

            temp = temp_init * (temp_decay ** it)

            for i in order:
                scores = np.zeros(k, dtype=np.float32)
                deg_i = max(len(pos_neighbors[i]) + len(neg_neighbors[i]), 1)

                if pos_weights[i] is not None:
                    for nbr, w in zip(pos_neighbors[i], pos_weights[i]):
                        scores[labels[nbr]] += alpha * float(w)/ deg_i
                else:
                    for nbr in pos_neighbors[i]:
                        scores[labels[nbr]] += alpha/ deg_i

                if neg_weights[i] is not None:
                    for nbr, w in zip(neg_neighbors[i], neg_weights[i]):
                        scores[labels[nbr]] -= alpha * neg_weight * float(abs(w))/ deg_i
                else:
                    for nbr in neg_neighbors[i]:
                        scores[labels[nbr]] -= alpha * neg_weight/ deg_i

                emb_i = embeddings[i]
                sim = cluster_centers @ emb_i
                scores += beta * sim

                                
                s = scores / max(1e-6, temp)
                s -= np.max(s)
                exp_s = np.exp(s)
                probs = exp_s / np.sum(exp_s)

                orig_cluster = labels[i]
                orig_size = np.sum(labels == orig_cluster)
                min_cluster_size = 3                 
                
                if stochastic:
                    candidates = np.arange(k)
                    valid_candidates = [c for c in candidates if c == orig_cluster or orig_size > min_cluster_size]
                    if len(valid_candidates) == 0:
                        new_label = orig_cluster
                    else:
                        probs_subset = probs[valid_candidates] / probs[valid_candidates].sum()
                        new_label = int(rng.choice(valid_candidates, p=probs_subset))
                else:
                    sorted_indices = np.argsort(-probs)
                    new_label = orig_cluster
                    for idx in sorted_indices:
                        candidate = idx
                        if candidate == orig_cluster or orig_size > min_cluster_size:
                            new_label = candidate
                            break

                if new_label != labels[i]:
                    old_label = labels[i]
                    labels[i] = new_label
                    changes += 1
                    refined_nodes[i] = (old_label, new_label)

                embeddings[i] = normalize(0.7*embeddings[i] + 0.3*cluster_centers[new_label].reshape(1,-1), axis=1)[0]

            if debug:
                print(f"[DenseIter] Iter {it+1}/{T}: {changes} label changes, temp={temp:.3f}")
            else:
                print(".", end='', flush=True)

            if changes == 0:
                break
        
        # Record refinement changes to tracker
        if tracker is not None:
            tracker.record_refinement(refined_nodes, labels_before, labels)

        return labels
