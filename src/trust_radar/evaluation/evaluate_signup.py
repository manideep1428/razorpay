"""Evaluation pipeline for the GraphSAGE Signup Trust Model."""

from pathlib import Path

import torch
from torch_geometric.data import Data

from trust_radar.models.signup_gnn import SignupGraphSAGE
from trust_radar.utils.metrics import (
    compute_classification_metrics,
    optimal_threshold_cost,
)
from trust_radar.utils.model_io import load_gnn_model


def evaluate_signup_gnn(
    model_or_path: SignupGraphSAGE | str | Path,
    data: Data,
    split: str = "test",
    device: torch.device | None = None,
) -> dict[str, float]:
    """Evaluate the Signup Trust Model on a specific graph split mask.

    Reports binary abuse-detection metrics plus a summary of the resulting
    0-100 trust-score distribution on the selected split.

    Args:
        model_or_path: Trained SignupGraphSAGE instance or checkpoint path.
        data: PyG Data object containing the requested split mask.
        split: Mask name ('test', 'val', or 'train').
        device: Device to run evaluation on.

    Returns:
        Dictionary of computed evaluation metrics.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(model_or_path, (str, Path)):
        model = load_gnn_model(model_or_path, device=dev)
    else:
        model = model_or_path.to(dev)

    model.eval()

    mask_attr = f"{split}_mask"
    if not hasattr(data, mask_attr):
        raise ValueError(f"Data object does not have mask: {mask_attr}")

    mask = getattr(data, mask_attr).to(dev)
    x = data.x.to(dev)
    edge_index = data.edge_index.to(dev)
    y = data.y.to(dev)

    with torch.no_grad():
        probs = model.predict_proba(x, edge_index).cpu().numpy().ravel()
        y_true = y.cpu().numpy().ravel()

    mask_np = mask.cpu().numpy()
    metrics = compute_classification_metrics(y_true[mask_np], probs[mask_np])
    best_thresh, min_cost = optimal_threshold_cost(y_true[mask_np], probs[mask_np])
    metrics["optimal_threshold"] = best_thresh
    metrics["min_cost"] = min_cost

    # Trust-score distribution on the split.
    scores = model.predict_scores(x, edge_index)
    trust_split = scores["trust_score"][mask_np]
    metrics["mean_trust_score"] = float(trust_split.mean())
    metrics["mean_risk_score"] = float(scores["risk_score"][mask_np].mean())

    return metrics
