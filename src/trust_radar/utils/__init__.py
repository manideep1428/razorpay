"""Utilities package for FraudShield AI."""

from trust_radar.utils.metrics import (
    compute_classification_metrics,
    compute_multiclass_metrics,
    multiclass_confusion,
    optimal_threshold_cost,
)
from trust_radar.utils.model_io import (
    load_gnn_model,
    load_tabular_model,
    save_gnn_model,
    save_tabular_model,
)
from trust_radar.utils.preprocessing import build_graph_data, prepare_tabular_features
from trust_radar.utils.synthetic import (
    synthesize_payment_dataset,
    synthesize_signup_dataset,
    synthesize_signup_edges,
)

__all__ = [
    "build_graph_data",
    "compute_classification_metrics",
    "compute_multiclass_metrics",
    "load_gnn_model",
    "load_tabular_model",
    "multiclass_confusion",
    "optimal_threshold_cost",
    "prepare_tabular_features",
    "save_gnn_model",
    "save_tabular_model",
    "synthesize_payment_dataset",
    "synthesize_signup_dataset",
    "synthesize_signup_edges",
]
