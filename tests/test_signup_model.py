"""Tests for the GraphSAGE Signup Trust Model: architecture, scores, inference."""

import numpy as np
import torch
from torch_geometric.data import Data

from trust_radar.config import FeatureConfig
from trust_radar.decisioning import RISK_LEVELS, SIGNUP_ACTIONS
from trust_radar.inference.predict_signup import SignupPredictor
from trust_radar.models.signup_gnn import SignupGraphSAGE
from trust_radar.utils.model_io import load_gnn_model, save_gnn_model
from trust_radar.utils.preprocessing import build_graph_data
from trust_radar.utils.synthetic import (
    synthesize_signup_dataset,
    synthesize_signup_edges,
)


def test_signup_graphsage_forward():
    num_nodes, in_channels = 50, 16
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 100))

    model = SignupGraphSAGE(in_channels=in_channels, hidden_channels=32, num_layers=2)

    logits = model(x, edge_index)
    assert logits.shape == (num_nodes, 1)

    probs = model.predict_proba(x, edge_index)
    assert probs.shape == (num_nodes,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_signup_predict_scores_are_0_100_and_complementary():
    num_nodes, in_channels = 40, 12
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 80))

    model = SignupGraphSAGE(in_channels=in_channels, hidden_channels=16)
    scores = model.predict_scores(x, edge_index)

    assert scores["risk_score"].shape == (num_nodes,)
    assert np.all(scores["risk_score"] >= 0) and np.all(scores["risk_score"] <= 100)
    # trust_score is exactly the complement of risk_score.
    assert np.all(scores["trust_score"] + scores["risk_score"] == 100)


def test_signup_predictor_emits_trust_risk_level_decision():
    cfg = FeatureConfig()
    df = synthesize_signup_dataset(n=120, seed=5)
    nodes = df[cfg.signup_numeric_features]
    edges = synthesize_signup_edges(len(df), avg_degree=4.0, seed=5)
    data = build_graph_data(nodes, edges, labels=df["label"])

    model = SignupGraphSAGE(in_channels=data.x.size(1), hidden_channels=16)
    predictor = SignupPredictor(model_or_path=model)

    result = predictor.predict_graph(data)
    assert len(result["trust_score"]) == len(df)
    assert np.all(result["risk_score"] >= 0) and np.all(result["risk_score"] <= 100)
    assert set(result["risk_level"]).issubset(set(RISK_LEVELS))
    assert set(result["decision"]).issubset(set(SIGNUP_ACTIONS))

    single = predictor.predict_single_node(3, data)
    assert 0 <= single["risk_score"] <= 100
    assert single["trust_score"] == 100 - single["risk_score"]
    assert single["risk_level"] in RISK_LEVELS
    assert single["decision"] in SIGNUP_ACTIONS


def test_signup_predictor_and_io(tmp_path):
    num_nodes, in_channels = 20, 4
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 30))
    data = Data(x=x, edge_index=edge_index)

    model = SignupGraphSAGE(in_channels=in_channels, hidden_channels=8, out_channels=1)
    ckpt_path = tmp_path / "model.pt"
    save_gnn_model(model, ckpt_path)
    loaded = load_gnn_model(ckpt_path)

    predictor = SignupPredictor(model_or_path=loaded)
    res = predictor.predict_graph(data)
    assert res["embeddings"].shape == (num_nodes, 8)
    assert len(res["trust_score"]) == num_nodes
