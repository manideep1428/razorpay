"""Training pipeline for the GraphSAGE Signup Trust Model."""

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from trust_radar.config import SignupGNNConfig
from trust_radar.models.signup_gnn import SignupGraphSAGE
from trust_radar.utils.metrics import compute_classification_metrics
from trust_radar.utils.model_io import save_gnn_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_signup_gnn(
    data: Data,
    config: SignupGNNConfig | None = None,
    device: torch.device | None = None,
) -> tuple[SignupGraphSAGE, dict[str, Any]]:
    """Train the GraphSAGE Signup Trust Model with early stopping.

    Args:
        data: PyG Data with x, edge_index, y, train_mask, val_mask, test_mask.
        config: Model and training hyper-parameters.
        device: Torch device (CPU or CUDA).

    Returns:
        (trained_model, history_dict)
    """
    cfg = config or SignupGNNConfig()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SignupGraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=cfg.hidden_channels,
        out_channels=cfg.out_channels,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(dev)

    x = data.x.to(dev)
    edge_index = data.edge_index.to(dev)
    y = data.y.to(dev)
    train_mask = data.train_mask.to(dev)
    val_mask = data.val_mask.to(dev)

    # Class weighting for abuse-class imbalance
    pos_weight = torch.tensor([cfg.pos_weight], device=dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    best_val_auc = -1.0
    best_state = None
    history: dict[str, Any] = {"train_loss": [], "val_auc": [], "val_pr_auc": []}

    logger.info("Starting Signup Trust Model training for %d epochs on %s...", cfg.epochs, dev)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index)
            val_probs = torch.sigmoid(val_logits[val_mask]).cpu().numpy().ravel()
            val_targets = y[val_mask].cpu().numpy().ravel()

            metrics = compute_classification_metrics(val_targets, val_probs)
            val_auc = metrics.get("roc_auc", 0.0)
            val_pr = metrics.get("pr_auc", 0.0)

        history["train_loss"].append(float(loss.item()))
        history["val_auc"].append(val_auc)
        history["val_pr_auc"].append(val_pr)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == cfg.epochs:
            logger.info(
                "Epoch %03d | Train Loss: %.4f | Val ROC-AUC: %.4f | Val PR-AUC: %.4f",
                epoch,
                loss.item(),
                val_auc,
                val_pr,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Report the resulting trust-score distribution on the validation split.
    scores = model.predict_scores(x, edge_index)
    val_np = val_mask.cpu().numpy()
    if val_np.any():
        logger.info(
            "Validation trust score -> mean: %.1f | min: %d | max: %d",
            float(np.mean(scores["trust_score"][val_np])),
            int(np.min(scores["trust_score"][val_np])),
            int(np.max(scores["trust_score"][val_np])),
        )

    save_gnn_model(model, cfg.checkpoint_path, metadata={"best_val_auc": best_val_auc})
    logger.info("Model checkpoint successfully saved to %s", cfg.checkpoint_path)

    return model, history


if __name__ == "__main__":
    # Synthetic self-test using the FraudShield signup schema.
    from trust_radar.config import FeatureConfig
    from trust_radar.utils.preprocessing import build_graph_data
    from trust_radar.utils.synthetic import (
        synthesize_signup_dataset,
        synthesize_signup_edges,
    )

    feat_cfg = FeatureConfig()
    df = synthesize_signup_dataset(n=800, seed=7)
    nodes = df[feat_cfg.signup_numeric_features]
    edges = synthesize_signup_edges(len(df), avg_degree=4.0, seed=7)
    demo_data = build_graph_data(nodes, edges, labels=df["label"])
    train_signup_gnn(demo_data, SignupGNNConfig(epochs=5))
