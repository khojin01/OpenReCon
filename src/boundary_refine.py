import torch
import numpy as np
from scipy.special import comb

class CommunityConstructor:
    def construct(self, clusters, A_p, A_n, tracker=None, **kwargs):
        print("\n>> Constructing partition using Boundary Refine (Fixed Initial Clusters + Purge & Move) strategy...")
        num_nodes = A_p.shape[0]
        clusters_before = np.copy(clusters)
        new_clusters = np.copy(clusters)

        lambda_purge = kwargs.get('lambda_purge', 0.8)
        s_min_import = kwargs.get('s_min_import', 0.5)
        p_min_import = kwargs.get('p_min_import', 0.8)
        print(f"    (params: lambda_purge={lambda_purge}, s_min_import={s_min_import}, p_min_import={p_min_import})")

        all_nodes_set = set(range(num_nodes))
        node_best_cluster = {}                
        node_best_score = {}
        all_purged_nodes = set()

        unique_clusters = np.unique(clusters)
        cluster_sets = [set(np.where(clusters == k)[0]) for k in unique_clusters]

        print(f"\n>>> Cluster count before Purge/Import: {len(cluster_sets)}")
        for idx, he in enumerate(cluster_sets):
            print(f"    Cluster {unique_clusters[idx]} size before Purge: {len(he)}")

        for idx, cluster_nodes in enumerate(cluster_sets):
            if len(cluster_nodes) < 3:
                continue

            cluster_id = unique_clusters[idx]
            print(f"\n--- Processing Cluster {cluster_id} (initial size: {len(cluster_nodes)}) ---")

            rho_scores = {}
            cluster_list = list(cluster_nodes)
            C_map = {node_id: i for i, node_id in enumerate(cluster_list)}

            A_p_C = A_p[cluster_list, :][:, cluster_list]
            friend_counts = A_p_C.sum(axis=1).A1
            enemy_counts = np.zeros(len(cluster_nodes))

            A_p_C_coo = A_p_C.tocoo()
            for u_idx, v_idx in zip(A_p_C_coo.row, A_p_C_coo.col):
                if u_idx >= v_idx:
                    continue
                u, v = cluster_list[u_idx], cluster_list[v_idx]
                u_neg = set(A_n.getrow(u).indices)
                v_neg = set(A_n.getrow(v).indices)
                common_enemies = u_neg.intersection(v_neg).intersection(cluster_nodes)
                for x in common_enemies:
                    enemy_counts[C_map[x]] += 1

            for i, node in enumerate(cluster_list):
                f, e = friend_counts[i], enemy_counts[i]
                
                internal_neg = 0
                neg_neighbors = set(A_n.getrow(node).indices)
                internal_neg = len(neg_neighbors.intersection(cluster_nodes))
                

                external_pos = 0
                pos_neighbors = set(A_p.getrow(node).indices)
                external_pos = len(pos_neighbors - cluster_nodes)
                
                rho_scores[node] = (e + internal_neg * 2 + external_pos * 0.5) / (f + e + internal_neg + external_pos + 1)

            rho_values = list(rho_scores.values())
            if len(rho_values) > 1:
                mu_rho = np.mean(rho_values)
                std_rho = np.std(rho_values)
                t_purge = mu_rho + lambda_purge * std_rho
            else:
                t_purge = 0
            
            nodes_to_purge = {node for node, rho in rho_scores.items() if rho > t_purge and rho > 0}
            
            for node in nodes_to_purge:
                new_clusters[node] = -1
                all_purged_nodes.add(node)
            cluster_nodes -= nodes_to_purge
            print(f">> Step 1: Purged {len(nodes_to_purge)} nodes. Remaining: {len(cluster_nodes)}")

            external_nodes = all_nodes_set - cluster_nodes
            for node in external_nodes:
                if new_clusters[node] != -1:                          
                    continue
                pos_neighbors_in_cluster = {n for n in A_p.getrow(node).indices if n in cluster_nodes}
                if len(pos_neighbors_in_cluster) < 2:
                    continue
                s_plus = 0
                pos_list = list(pos_neighbors_in_cluster)
                for i in range(len(pos_list)):
                    for j in range(i + 1, len(pos_list)):
                        u, v = pos_list[i], pos_list[j]
                        if A_p[u, v] > 0:
                            s_plus += 1
                possible_triangles = comb(len(pos_neighbors_in_cluster), 2)
                p_plus = s_plus / possible_triangles if possible_triangles > 0 else 0.0

                if p_plus >= max(s_min_import, p_min_import):
                    if node not in node_best_score or p_plus > node_best_score[node]:
                        node_best_score[node] = p_plus
                        node_best_cluster[node] = cluster_id

        imported_nodes = {}
        for node, cluster_id in node_best_cluster.items():
            if new_clusters[node] == -1:                           
                new_clusters[node] = cluster_id
                cluster_sets[unique_clusters.tolist().index(cluster_id)].add(node)
                imported_nodes[node] = cluster_id

        print(f">> Step 2: Moved {sum(new_clusters != clusters_before)} nodes into clusters.")
        
        unassigned_nodes = np.where(new_clusters == -1)[0]
        if len(unassigned_nodes) > 0:
            
            for node in unassigned_nodes:
                best_cluster = None
                best_score = -float('inf')
                
                for idx, cluster_nodes in enumerate(cluster_sets):
                    if len(cluster_nodes) == 0:
                        continue
                    
                    cluster_id = unique_clusters[idx]
                    
                    pos_connections = sum(1 for c_node in cluster_nodes if A_p[node, c_node] > 0)
                    neg_connections = sum(1 for c_node in cluster_nodes if A_n[node, c_node] > 0)
                    
                    score = pos_connections - neg_connections
                    
                    if score > best_score:
                        best_score = score
                        best_cluster = (cluster_id, idx)
                
                if best_cluster is not None and best_score > 0:
                    cluster_id, idx = best_cluster
                    new_clusters[node] = cluster_id
                    cluster_sets[idx].add(node)
                    imported_nodes[node] = cluster_id
                else:
                    next_cluster_id = max(unique_clusters) + 1
                    new_clusters[node] = next_cluster_id
                    cluster_sets.append({node})
                    unique_clusters = np.append(unique_clusters, next_cluster_id)
                    imported_nodes[node] = next_cluster_id

        # Record purge and import to tracker
        if tracker is not None:
            all_purged = set(np.where(new_clusters == -1)[0]) | all_purged_nodes
            tracker.record_purge(all_purged, clusters_before, new_clusters)
            tracker.record_import(imported_nodes, new_clusters)

        final_communities = [frozenset(h) for h in cluster_sets if len(h) > 0]

        if not final_communities:
            return torch.LongTensor([[], []]), num_nodes, 0, new_clusters, []

        node_indices = [node for he in final_communities for node in he]
        edge_indices = [i for i, he in enumerate(final_communities) for _ in he]

        cluster_index = torch.LongTensor([node_indices, edge_indices])
        num_clusters_out = len(final_communities)

        return cluster_index, num_nodes, num_clusters_out, new_clusters, final_communities
