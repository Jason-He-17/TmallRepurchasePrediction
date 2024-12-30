# pytorch_hetero_graph_feature.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import pandas as pd

from helper import _cap_values, _bin_values  # 确保 helper.py 中定义了这些函数

def add_pytorch_hetero_graph_embedding_features(matrix, origin_data, embed_dim=16, epochs=170, bins=16, lr=0.01):
    """
    利用异质图结构学习更高阶的用户-商家-类目-品牌-商品等多节点关系，并将嵌入特征合并到 matrix 中。
    
    :param matrix: 训练/测试数据矩阵对象，包含 train_test_matrix 属性
    :param origin_data: 原始数据对象，包含 user_log_format1 属性
    :param embed_dim: 嵌入向量维度
    :param epochs: 训练轮数
    :param lr: 学习率
    """

    user_log = origin_data.user_log_format1.copy()

    # 1) 收集不同类型节点的映射
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

    print("节点映射已完成。")

    # 2) 构建 HeteroData
    data = HeteroData()

    # 加入节点特征 (此处使用随机向量示例，实际应用中应使用真实特征)
    data['user'].x = torch.randn(len(user_map), embed_dim)
    data['merchant'].x = torch.randn(len(merchant_map), embed_dim)
    data['category'].x = torch.randn(len(cat_map), embed_dim)
    data['brand'].x = torch.randn(len(brand_map), embed_dim)
    data['item'].x = torch.randn(len(item_map), embed_dim)

    print("节点特征已初始化。")

    # 3) 构建异质边 (User->Merchant)
    u_indices = user_log['user_id'].map(user_map).astype(int).tolist()
    m_indices = user_log['merchant_id'].map(merchant_map).astype(int).tolist()
    data['user', 'buys', 'merchant'].edge_index = torch.tensor([u_indices, m_indices], dtype=torch.long)

    print("User->Merchant 边已添加。")

    # 4) 构建异质边 (Merchant->Category)
    mc_group = user_log[['merchant_id', 'cat_id']].drop_duplicates()
    mc_m = mc_group['merchant_id'].map(merchant_map).astype(int).tolist()
    mc_c = mc_group['cat_id'].map(cat_map).astype(int).tolist()
    data['merchant', 'has', 'category'].edge_index = torch.tensor([mc_m, mc_c], dtype=torch.long)

    print("Merchant->Category 边已添加。")

    # 5) 构建异质边 (Merchant->Brand)
    mb_group = user_log[['merchant_id', 'brand_id']].drop_duplicates()
    mb_m = mb_group['merchant_id'].map(merchant_map).astype(int).tolist()
    mb_b = mb_group['brand_id'].map(brand_map).astype(int).tolist()
    data['merchant', 'has', 'brand'].edge_index = torch.tensor([mb_m, mb_b], dtype=torch.long)

    print("Merchant->Brand 边已添加。")

    # 6) 构建异质边 (Merchant->Item)
    mi_group = user_log[['merchant_id', 'item_id']].drop_duplicates()
    mi_m = mi_group['merchant_id'].map(merchant_map).astype(int).tolist()
    mi_i = mi_group['item_id'].map(item_map).astype(int).tolist()
    data['merchant', 'contains', 'item'].edge_index = torch.tensor([mi_m, mi_i], dtype=torch.long)

    print("Merchant->Item 边已添加。")

    # 7) 添加自连接边 (Self-Loops) 以确保所有节点类型都有接收边
    # 这里只为 'user' 添加自连接边
    user_self_edges = torch.stack([torch.arange(len(user_map)), torch.arange(len(user_map))], dim=0)
    data['user', 'self', 'user'].edge_index = user_self_edges

    print("User->User 自连接边已添加。")

    # 确认所有边类型是否已添加
    print("所有边类型已添加：", data.edge_types)

    # 8) 定义异质图模型
    class HeteroGNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = HeteroConv({
                ('user', 'buys', 'merchant'): SAGEConv(in_channels, hidden_channels),
                ('merchant', 'has', 'category'): SAGEConv(in_channels, hidden_channels),
                ('merchant', 'has', 'brand'): SAGEConv(in_channels, hidden_channels),
                ('merchant', 'contains', 'item'): SAGEConv(in_channels, hidden_channels),
                ('user', 'self', 'user'): SAGEConv(in_channels, hidden_channels),  # 自连接边
            }, aggr='mean')
            
            self.conv2 = HeteroConv({
                ('user', 'buys', 'merchant'): SAGEConv(hidden_channels, out_channels),
                ('merchant', 'has', 'category'): SAGEConv(hidden_channels, out_channels),
                ('merchant', 'has', 'brand'): SAGEConv(hidden_channels, out_channels),
                ('merchant', 'contains', 'item'): SAGEConv(hidden_channels, out_channels),
                ('user', 'self', 'user'): SAGEConv(hidden_channels, out_channels),  # 自连接边
            }, aggr='mean')

        def forward(self, x_dict, edge_index_dict):
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = self.conv2(x_dict, edge_index_dict)
            return x_dict

    print("异质图模型已定义。")

    # 9) 设置设备 (GPU 优先，如果可用)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Hetero Graph device: {device}')

    data = data.to(device)
    model = HeteroGNN(embed_dim, embed_dim, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("模型和优化器已初始化。")

    # 10) 训练模型
    print("开始训练模型...")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        try:
            out_dict = model(data.x_dict, data.edge_index_dict)
        except AttributeError as e:
            print(f"错误在模型的前向传播: {e}")
            return  # 或者根据需求进行处理

        # 自监督重构损失
        loss = 0
        for ntype in out_dict:
            if out_dict[ntype] is not None:
                loss += F.mse_loss(out_dict[ntype], data[ntype].x)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    print("模型训练完成。")

    # 11) 获取节点嵌入
    model.eval()
    with torch.no_grad():
        embed_dict = model(data.x_dict, data.edge_index_dict)

    print("节点嵌入已生成。")

    # 12) 将嵌入转换为 DataFrame 并合并回 matrix
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

    print("嵌入 DataFrame 已生成。")

    # 13) 合并嵌入特征回矩阵
    matrix.train_test_matrix = matrix.train_test_matrix.merge(user_embed_df, on='user_id', how='left')
    matrix.train_test_matrix = matrix.train_test_matrix.merge(merchant_embed_df, on='merchant_id', how='left')

    print("嵌入特征已合并回矩阵。")

    # 14) 对嵌入做分位截断和分箱
    all_emb_cols = [c for c in matrix.train_test_matrix.columns if c.startswith(('user_emb', 'merchant_emb'))]
    for col in all_emb_cols:
        matrix.train_test_matrix[col] = _cap_values(matrix.train_test_matrix[col], upper_percentile=95)
        matrix.train_test_matrix[col] = _bin_values(matrix.train_test_matrix[col], bins)

    print("嵌入特征已进行截断和分箱。")

    # 15) 缺失值填充
    matrix.train_test_matrix[all_emb_cols] = matrix.train_test_matrix[all_emb_cols].fillna(0)

    print("缺失值已填充。")
    print("Heterogeneous graph embedding features added.")

    # 清理cuda内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()