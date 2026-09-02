"""Inference engine for the GraphSAGE Signup Trust Model."""

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from trust_radar.decisioning import (
    risk_level_from_score,
    risk_score_to_trust_score,
    signup_decision,
)
from trust_radar.models.signup_gnn import SignupGraphSAGE
from trust_radar.utils.model_io import load_gnn_model


class SignupPredictor:
    """Production inference for signup trust scoring using GraphSAGE.

    Emits, per node, an ``abuse_probability``, a 0-100 ``risk_score`` and
    complementary ``trust_score``, a ``risk_level`` (low/medium/high/critical),
    and the 4-tier signup ``decision``.
    """

    def __init__(
        self,
        model_or_path: SignupGraphSAGE | str | Path,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if isinstance(model_or_path, (str, Path)):
            self.model = load_gnn_model(model_or_path, device=self.device)
        else:
            self.model = model_or_path.to(self.device)
        self.model.eval()

    def predict_graph(self, data: Data) -> dict[str, np.ndarray]:
        """Run inference over all nodes in the graph.

        Returns arrays for probabilities, risk_scores, trust_scores,
        risk_levels, decisions, and node embeddings.
        """
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        with torch.no_grad():
            embeddings = self.model.forward_embeddings(x, edge_index).cpu().numpy()

        scores = self.model.predict_scores(x, edge_index)
        risk_scores = scores["risk_score"]

        risk_levels = np.array([risk_level_from_score(s) for s in risk_scores])
        decisions = np.array([signup_decision(s) for s in risk_scores])

        return {
            "abuse_probability": scores["abuse_probability"],
            "risk_score": risk_scores,
            "trust_score": scores["trust_score"],
            "risk_level": risk_levels,
            "decision": decisions,
            "embeddings": embeddings,
        }

    def predict_single_node(
        self, node_idx: int, data: Data
    ) -> dict[str, float | int | str | np.ndarray]:
        """Return the full trust assessment for a single node index."""
        results = self.predict_graph(data)
        risk_score = int(results["risk_score"][node_idx])
        return {
            "node_idx": node_idx,
            "abuse_probability": float(results["abuse_probability"][node_idx]),
            "risk_score": risk_score,
            "trust_score": risk_score_to_trust_score(risk_score),
            "risk_level": str(results["risk_level"][node_idx]),
            "decision": str(results["decision"][node_idx]),
            "embedding": results["embeddings"][node_idx],
        }
