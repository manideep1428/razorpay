"""GraphSAGE Signup Trust Model for FraudShield AI.

The model detects fake signups, multi-account abuse, bot signups, device
farming, VPN abuse, trial farmers, and account-creation rings by propagating
information across shared device, IP, phone, and email-domain networks.

It is a binary classifier (``0 = legit_user``, ``1 = abuse_user``) whose
probability output is converted into a 0-100 ``risk_score`` and its complement
``trust_score`` via :mod:`trust_radar.decisioning`.
"""


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class SignupGraphSAGE(nn.Module):
    """GraphSAGE network for detecting fraudulent and abusive signups."""

    def __init__(
        self,
        in_channels: int = 32,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        # Final convolution / embedding projection
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Extract node representation embeddings."""
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits for each node.

        Args:
            x: Node feature matrix of shape (num_nodes, in_channels).
            edge_index: Graph connectivity of shape (2, num_edges).

        Returns:
            Logits of shape (num_nodes, out_channels).
        """
        emb = self.forward_embeddings(x, edge_index)
        logits = self.classifier(emb)
        return logits

    def predict_proba(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute predicted probability of signup abuse for each node.

        Returns:
            Probability tensor of shape (num_nodes,).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            if self.out_channels == 1:
                return torch.sigmoid(logits).squeeze(-1)
            return F.softmax(logits, dim=-1)

    def predict_scores(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, np.ndarray]:
        """Compute FraudShield trust/risk scores for every node.

        Returns a dict with:
            * ``abuse_probability`` -- raw 0-1 abuse probability.
            * ``risk_score``        -- integer 0-100 (higher = riskier).
            * ``trust_score``       -- integer 0-100 (100 - risk_score).
        """
        probs = self.predict_proba(x, edge_index).cpu().numpy().ravel()
        probs = np.clip(probs, 0.0, 1.0)
        risk_scores = np.rint(probs * 100).astype(int)
        trust_scores = 100 - risk_scores
        return {
            "abuse_probability": probs,
            "risk_score": risk_scores,
            "trust_score": trust_scores,
        }
