# pytorch_hetero_graph_feature.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import pandas as pd

from helper import _cap_values, _bin_values  # Ensure these functions are defined in helper.py

def add_pytorch_hetero_graph_embedding_features(matrix, origin_data, embed_dim=16, epochs=170, bins=16, lr=0.01):
    """
    Learn higher-order relationships among users, merchants, categories, brands, items using a heterogeneous graph structure,
    and merge the embedding features into matrix.
    
    :param matrix: Training/testing data matrix object, which has the train_test_matrix attribute
    :param origin_data: Original data object, which has the user_log_format1 attribute
    :param embed_dim: Embedding vector dimension
    :param epochs: Number of training epochs
    :param lr: Learning rate
    """

    user_log = origin_data.user_log_format1.copy()

    # 1) Collect mappings for different types of nodes
    user_ids = user_log['user_id'].unique()
    merchant_ids = user_log['merchant_id'].unique()
    cat_ids = user_log['cat_id'].unique()
    brand_ids = user_log['brand_id'].unique()
    item_ids = user_log['item_id'].unique()

    user_map = {uid: i for i, uid in enumerate(user_ids)}
    merchant_map = {mid: i for i, mid in enumerate(merchant_ids)}
    cat_map = {cid: i for i, cid in enumerate(cat_ids)}
    brand_map = {bid: i for i, bid in enumerate(brand_ids)}
    item_map = {iid: i for i, iid in enumerate(item_ids)}

    print("Node mapping completed.")

    # 2) Build HeteroData
    data = HeteroData()

    # Add node features (using random vectors here as an example; in real applications, use actual features)
    data['user'].x = torch.randn(len(user_map), embed_dim)
    data['merchant'].x = torch.randn(len(merchant_map), embed_dim)
    data['category'].x = torch.randn(len(cat_map), embed_dim)
    data['brand'].x = torch.randn(len(brand_map), embed_dim)
    data['item'].x = torch.randn(len(item_map), embed_dim)

    print("Node features initialized.")

    # 3) Build heterogeneous edges (User->Merchant)
    u_indices = user_log['user_id'].map(user_map).astype(int).tolist()
    m_indices = user_log['merchant_id'].map(merchant_map).astype(int).tolist()
    data['user', 'buys', 'merchant'].edge_index = torch.tensor([u_indices, m_indices], dtype=torch.long)

    print("User->Merchant edge added.")

    # 4) Build heterogeneous edges (Merchant->Category)
    mc_group = user_log[['merchant_id', 'cat_id']].drop_duplicates()
    mc_m = mc_group['merchant_id'].map(merchant_map).astype(int).tolist()
    mc_c = mc_group['cat_id'].map(cat_map).astype(int).tolist()
    data['merchant', 'has', 'category'].edge_index = torch.tensor([mc_m, mc_c], dtype=torch.long)

    print("Merchant->Category edge added.")

    # 5) Build heterogeneous edges (Merchant->Brand)
    mb_group = user_log[['merchant_id', 'brand_id']].drop_duplicates()
    mb_m = mb_group['merchant_id'].map(merchant_map).astype(int).tolist()
    mb_b = mb_group['brand_id'].map(brand_map).astype(int).tolist()
    data['merchant', 'has', 'brand'].edge_index = torch.tensor([mb_m, mb_b], dtype=torch.long)

    print("Merchant->Brand edge added.")

    # 6) Build heterogeneous edges (Merchant->Item)
    mi_group = user_log[['merchant_id', 'item_id']].drop_duplicates()
    mi_m = mi_group['merchant_id'].map(merchant_map).astype(int).tolist()
    mi_i = mi_group['item_id'].map(item_map).astype(int).tolist()
    data['merchant', 'contains', 'item'].edge_index = torch.tensor([mi_m, mi_i], dtype=torch.long)

    print("Merchant->Item edge added.")

    # 7) Add self-loops edges (Self-Loops) to ensure all node types have receiving edges
    # Here we only add self-loops for 'user'
    user_self_edges = torch.stack([torch.arange(len(user_map)), torch.arange(len(user_map))], dim=0)
    data['user', 'self', 'user'].edge_index = user_self_edges

    print("User->User self-loop edge added.")

    # Confirm all edge types have been added
    print("All edge types added:", data.edge_types)

    # 8) Define the heterogeneous graph model
    class HeteroGNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = HeteroConv({
                ('user', 'buys', 'merchant'): SAGEConv(in_channels, hidden_channels),
                ('merchant', 'has', 'category'): SAGEConv(in_channels, hidden_channels),
                ('merchant', 'has', 'brand'): SAGEConv(in_channels, hidden_channels),
                ('merchant', 'contains', 'item'): SAGEConv(in_channels, hidden_channels),
                ('user', 'self', 'user'): SAGEConv(in_channels, hidden_channels),  # Self-loops
            }, aggr='mean')
            
            self.conv2 = HeteroConv({
                ('user', 'buys', 'merchant'): SAGEConv(hidden_channels, out_channels),
                ('merchant', 'has', 'category'): SAGEConv(hidden_channels, out_channels),
                ('merchant', 'has', 'brand'): SAGEConv(hidden_channels, out_channels),
                ('merchant', 'contains', 'item'): SAGEConv(hidden_channels, out_channels),
                ('user', 'self', 'user'): SAGEConv(hidden_channels, out_channels),  # Self-loops
            }, aggr='mean')

        def forward(self, x_dict, edge_index_dict):
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = self.conv2(x_dict, edge_index_dict)
            return x_dict

    print("Heterogeneous graph model defined.")

    # 9) Set device (prefer GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Hetero Graph device: {device}')

    data = data.to(device)
    model = HeteroGNN(embed_dim, embed_dim, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Model and optimizer initialized.")

    # 10) Train the model
    print("Starting model training...")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        try:
            out_dict = model(data.x_dict, data.edge_index_dict)
        except AttributeError as e:
            print(f"Error during model forward pass: {e}")
            return  # Or handle the error as needed

        # Self-supervised reconstruction loss
        loss = 0
        for ntype in out_dict:
            if out_dict[ntype] is not None:
                loss += F.mse_loss(out_dict[ntype], data[ntype].x)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    print("Model training complete.")

    # 11) Get node embeddings
    model.eval()
    with torch.no_grad():
        embed_dict = model(data.x_dict, data.edge_index_dict)

    print("Node embeddings generated.")

    # 12) Convert embeddings to DataFrame and merge back into matrix
    def to_df(embed_tensor, index_map, col_prefix):
        arr = embed_tensor.cpu().numpy()
        inv_map = {v: k for k, v in index_map.items()}
        df = pd.DataFrame(arr, index=[inv_map[i] for i in range(len(inv_map))])
        df.index.name = f'{col_prefix}_id'
        df.reset_index(inplace=True)
        df.columns = [f'{col_prefix}_id'] + [f'{col_prefix}_emb_{i}' for i in range(embed_dim)]
        return df

    user_embed_df = to_df(embed_dict['user'], user_map, 'user')
    merchant_embed_df = to_df(embed_dict['merchant'], merchant_map, 'merchant')
    category_embed_df = to_df(embed_dict['category'], cat_map, 'category')
    brand_embed_df = to_df(embed_dict['brand'], brand_map, 'brand')
    item_embed_df = to_df(embed_dict['item'], item_map, 'item')

    print("Embedding DataFrames generated.")

    # 13) Merge embedding features back into the matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(user_embed_df, on='user_id', how='left')
    matrix.train_test_matrix = matrix.train_test_matrix.merge(merchant_embed_df, on='merchant_id', how='left')

    print("Embedding features merged back into the matrix.")

    # 14) Apply percentile capping and binning to embeddings
    all_emb_cols = [c for c in matrix.train_test_matrix.columns if c.startswith(('user_emb', 'merchant_emb'))]
    for col in all_emb_cols:
        matrix.train_test_matrix[col] = _cap_values(matrix.train_test_matrix[col], upper_percentile=95)
        matrix.train_test_matrix[col] = _bin_values(matrix.train_test_matrix[col], bins)

    print("Embedding features capped and binned.")

    # 15) Fill missing values
    matrix.train_test_matrix[all_emb_cols] = matrix.train_test_matrix[all_emb_cols].fillna(0)

    print("Missing values filled.")
    print("Heterogeneous graph embedding features added.")

    # Clear CUDA memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()