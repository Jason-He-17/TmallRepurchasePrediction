# pytorch_graph_feature.py

from helper import _cap_values, _bin_values

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np

def add_pytorch_graph_embedding_features(matrix, origin_data, embed_dim=16, feature_dim=16, cap_percentile=95, bins=16, epochs=100, lr=0.01):
    """
    Generate graph embedding features for user and merchant nodes using PyTorch Geometric and merge them into matrix.
    
    :param matrix: Training/testing data matrix object
    :param origin_data: Original data object containing user_log_format1
    :param embed_dim: Embedding vector dimension (set to match feature_dim)
    :param feature_dim: Initial node feature dimension
    :param cap_percentile: Percentile cap value
    :param bins: Number of bins for binning
    :param epochs: Number of training epochs
    :param lr: Learning rate
    """

    user_log = origin_data.user_log_format1.copy()
    
    # 1) Prepare bipartite graph node mapping
    users = user_log['user_id'].unique()
    merchants = user_log['merchant_id'].unique()

    user_map = {uid: i for i, uid in enumerate(users)}
    merchant_map = {mid: i + len(users) for i, mid in enumerate(merchants)}

    # 2) Build edge set
    edges_u = user_log['user_id'].map(user_map).values
    edges_m = user_log['merchant_id'].map(merchant_map).values
    edge_index = torch.tensor(np.array([edges_u, edges_m]), dtype=torch.long)  # Optimize edge list conversion

    # 3) Build graph data (using randomly initialized node features)
    num_nodes = len(users) + len(merchants)
    x = torch.randn(num_nodes, feature_dim)
    data = Data(x=x, edge_index=edge_index)

    # 4) Set device (prefer GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Move data to device
    data = data.to(device)

    # 5) Define a simple GraphSAGE model
    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x

    model = GraphSAGE(data.num_features, embed_dim * 2, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 6) Simple self-supervised training (adjust loss function to match dimensions)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        # Use an autoencoder-style reconstruction loss, reconstructing only feature_dim dimensions
        # Ensure out and data.x are on the same device
        loss = F.mse_loss(out[:, :feature_dim], data.x)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # 7) Get post-training node embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data).cpu().numpy()

    # 8) Split embeddings for user and merchant nodes
    user_embeddings = {uid: embeddings[user_map[uid]] for uid in users}
    merchant_embeddings = {mid: embeddings[merchant_map[mid]] for mid in merchants}

    # 9) Convert to DataFrame and merge into matrix
    user_emb_df = pd.DataFrame.from_dict(user_embeddings, orient='index')
    user_emb_df.index.name = 'user_id'
    user_emb_df.reset_index(inplace=True)
    user_emb_df.columns = ['user_id'] + [f'u_emb_{i}' for i in range(embed_dim)]

    merchant_emb_df = pd.DataFrame.from_dict(merchant_embeddings, orient='index')
    merchant_emb_df.index.name = 'merchant_id'
    merchant_emb_df.reset_index(inplace=True)
    merchant_emb_df.columns = ['merchant_id'] + [f'm_emb_{i}' for i in range(embed_dim)]

    matrix.train_test_matrix = matrix.train_test_matrix.merge(user_emb_df, on='user_id', how='left')
    matrix.train_test_matrix = matrix.train_test_matrix.merge(merchant_emb_df, on='merchant_id', how='left')

    # 10) Apply percentile capping and binning to embeddings
    all_emb_cols = [col for col in matrix.train_test_matrix.columns if col.startswith(('u_emb_', 'm_emb_'))]
    for col in all_emb_cols:
        matrix.train_test_matrix[col] = _cap_values(matrix.train_test_matrix[col], upper_percentile=cap_percentile)
        matrix.train_test_matrix[col] = _bin_values(matrix.train_test_matrix[col], bins=bins)

    # 11) Fill missing values
    emb_cols_to_fill = [col for col in all_emb_cols if col in matrix.train_test_matrix.columns]
    matrix.train_test_matrix[emb_cols_to_fill] = matrix.train_test_matrix[emb_cols_to_fill].fillna(0)

    # Clear cuda memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()