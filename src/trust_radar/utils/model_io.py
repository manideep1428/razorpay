"""Model persistence utilities for GNNs and Tabular estimators."""

from pathlib import Path
from typing import Any

import joblib
import torch

from trust_radar.models.payment_model import PaymentAbuseModel
from trust_radar.models.signup_gnn import SignupGraphSAGE


def save_gnn_model(
    model: SignupGraphSAGE,
    filepath: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save PyTorch GNN model weights, architecture parameters, and metadata."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "state_dict": model.state_dict(),
        "in_channels": model.in_channels,
        "hidden_channels": model.hidden_channels,
        "out_channels": model.out_channels,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "metadata": metadata or {},
    }
    torch.save(state, path)
    return path


def load_gnn_model(
    filepath: str | Path,
    device: torch.device | None = None,
) -> SignupGraphSAGE:
    """Load PyTorch GNN model from checkpoint file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No GNN checkpoint found at: {path}")

    checkpoint = torch.load(path, map_location=device or torch.device("cpu"))
    model = SignupGraphSAGE(
        in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint["hidden_channels"],
        out_channels=checkpoint["out_channels"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    if device:
        model.to(device)
    model.eval()
    return model


def save_tabular_model(
    model: PaymentAbuseModel,
    filepath: str | Path,
) -> Path:
    """Save PaymentAbuseModel artifact using joblib."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_tabular_model(filepath: str | Path) -> PaymentAbuseModel:
    """Load PaymentAbuseModel artifact from joblib file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No tabular model found at: {path}")
    model: PaymentAbuseModel = joblib.load(path)
    return model
