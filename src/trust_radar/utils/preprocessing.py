"""Data preprocessing, tabular transformation, and graph structure construction utilities."""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def build_graph_data(
    node_features: pd.DataFrame,
    edges: pd.DataFrame,
    labels: pd.Series | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> Data:
    """
    Construct a PyTorch Geometric Data object from dataframes.

    Args:
        node_features: DataFrame of node attributes indexed by node_id (or 0..N).
        edges: DataFrame with columns ['src', 'dst'] or ['source', 'target'].
        labels: Optional binary target series.
        train_ratio: Proportion of nodes for training mask.
        val_ratio: Proportion of nodes for validation mask.
        random_state: Random seed for split masks.

    Returns:
        torch_geometric.data.Data object with x, edge_index, y, train_mask, val_mask, test_mask.
    """
    # Normalize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(node_features.values)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    # Edge index
    src_col = "src" if "src" in edges.columns else "source"
    dst_col = "dst" if "dst" in edges.columns else "target"
    edge_src = edges[src_col].values
    edge_dst = edges[dst_col].values

    # Undirected / bidirectional edges
    edge_index_np = np.vstack([
        np.concatenate([edge_src, edge_dst]),
        np.concatenate([edge_dst, edge_src]),
    ])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    # Labels
    y_tensor = None
    if labels is not None:
        y_tensor = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)

    # Masks
    num_nodes = len(node_features)
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(num_nodes)

    train_end = int(num_nodes * train_ratio)
    val_end = int(num_nodes * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return data


def prepare_tabular_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str | None = None,
    impute_strategy: str = "median",
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Clean, impute, and extract features for tabular models (LightGBM).

    Args:
        df: Input raw tabular dataframe.
        feature_cols: List of column names to keep.
        target_col: Optional name of the binary target label.
        impute_strategy: Strategy for missing value handling.

    Returns:
        (X_features, y_target)
    """
    X = df[feature_cols].copy()

    # Fill numerical missing values
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].isnull().any():
            if impute_strategy == "median":
                fill_val = X[col].median()
            else:
                fill_val = X[col].mean()
            X[col] = X[col].fillna(fill_val)

    # Fill categorical missing values
    for col in X.select_dtypes(exclude=[np.number]).columns:
        X[col] = X[col].fillna("UNKNOWN").astype("category")

    y = df[target_col] if target_col and target_col in df.columns else None
    return X, y
