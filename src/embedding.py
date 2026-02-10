import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cl_models import ClusterEncoder, CL
from cl_utils import drop_features, drop_incidence

def run_cl_embedding(features, cluster_index, num_nodes, num_clusters, device='cpu', epochs=100, hid_dim=128, proj_dim=128):
    
    # 1. Prepare inputs
    features = torch.FloatTensor(features).to(device)
    cluster_index = cluster_index.to(device)

    if num_clusters == 0:
        print("Warning: No cluster provided. Skipping CL training.")
        return features.cpu().numpy()

    print(f"Adapting contrastive learning with {num_clusters} clusters for {num_nodes} nodes.")

    # 2. Initialize model and optimizer
    encoder = ClusterEncoder(features.shape[1], hid_dim, hid_dim, num_layers=2)
    model = CL(encoder, proj_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    params = {
        'drop_incidence_rate': 0.1,
        'drop_feature_rate': 0.1,
        'tau_n': 0.5,
        'tau_g': 0.5,
        'tau_m': 1.0,
        'w_g': 1.0,
        'w_m': 1.0,
        'batch_size_1': None,
        'batch_size_2': None,
    }

    # 3. Training loop
    print(f"Training with contrastive learning for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Data Augmentation
        cluster_index1 = drop_incidence(cluster_index, params['drop_incidence_rate'])
        cluster_index2 = drop_incidence(cluster_index, params['drop_incidence_rate'])
        x1 = drop_features(features, params['drop_feature_rate'])
        x2 = drop_features(features, params['drop_feature_rate'])

        # Encoder
        n1, e1 = model(x1, cluster_index1, num_nodes, num_clusters)
        n2, e2 = model(x2, cluster_index2, num_nodes, num_clusters)

        # Projection
        n1, n2 = model.node_projection(n1), model.node_projection(n2)
        e1, e2 = model.edge_projection(e1), model.edge_projection(e2)

        # Loss calculation
        loss_n = model.node_level_loss(n1, n2, params['tau_n'], batch_size=params['batch_size_1'])
        loss_g = model.group_level_loss(e1, e2, params['tau_g'], batch_size=params['batch_size_1'])
        loss_m1 = model.membership_level_loss(n1, e2, cluster_index2, params['tau_m'], batch_size=params['batch_size_2'])
        loss_m2 = model.membership_level_loss(n2, e1, cluster_index1, params['tau_m'], batch_size=params['batch_size_2'])
        loss_m = (loss_m1 + loss_m2) * 0.5
        
        loss = loss_n + params['w_g'] * loss_g + params['w_m'] * loss_m
        
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d}, Loss: {loss.item():.4f}")

    # 4. Get final embeddings
    model.eval()
    with torch.no_grad():
        node_embeddings, _ = model(features, cluster_index)
    
    print("contrastive learning complete.")
    return node_embeddings.cpu().numpy()