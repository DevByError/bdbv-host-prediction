"""
Bipartite virus-host GNN, edge-level susceptibility prediction.

Not part of TrainingPipeline (different data shape); run it via
scripts/run_gnn_training.py. torch/torch_geometric are optional - the
import stays soft so src.models imports fine without them.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GNNModel:
    """GCN encoder over the virus/host graph plus an MLP edge classifier."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        device: str = "cpu",
        random_state: int = 42,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch and torch_geometric are required for GNNModel. "
                "Install with: pip install torch torch-geometric"
            )
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.random_state = random_state

        torch.manual_seed(random_state)

        self.model = None
        self.node_features = None
        self.edge_index = None
        self.virus_names: List[str] = []
        self.host_names: List[str] = []
        self.n_viruses = 0
        self.n_hosts = 0
        self._is_fitted = False
        self.training_history: Dict = {}

    def build_graph(
        self,
        virus_features: np.ndarray,
        host_features: np.ndarray,
        interactions: List[Tuple[int, int, float]],
        virus_names: List[str],
        host_names: List[str],
    ) -> "Data":
        """
        interactions are (virus_idx, host_idx, label) triples.

        Virus and host features get stacked into one node matrix, so they
        have to be the same width - pad or project beforehand.
        """
        if virus_features.shape[1] != host_features.shape[1]:
            raise ValueError(
                f"virus_features dim ({virus_features.shape[1]}) must match "
                f"host_features dim ({host_features.shape[1]}); project onto "
                f"a shared feature space first"
            )
        if not interactions:
            raise ValueError("interactions must not be empty")

        n_viruses = len(virus_features)
        n_hosts = len(host_features)

        features = np.vstack([virus_features, host_features])
        features = torch.tensor(features, dtype=torch.float)

        edges = []
        edge_labels = []

        for v_idx, h_idx, label in interactions:
            edges.append([v_idx, n_viruses + h_idx])
            edge_labels.append(label)
            edges.append([n_viruses + h_idx, v_idx])  # undirected
            edge_labels.append(label)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_labels = torch.tensor(edge_labels, dtype=torch.float)

        data = Data(x=features, edge_index=edge_index, edge_attr=edge_labels.unsqueeze(1))

        self.node_features = features
        self.edge_index = edge_index
        self.virus_names = virus_names
        self.host_names = host_names
        self.n_viruses = n_viruses
        self.n_hosts = n_hosts

        return data

    def build_model(self, input_dim: int) -> "nn.Module":
        class HostVirusGNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                super().__init__()

                self.convs = nn.ModuleList()
                self.bns = nn.ModuleList()

                self.convs.append(GCNConv(input_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

                for _ in range(num_layers - 2):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                    self.bns.append(nn.BatchNorm1d(hidden_dim))

                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.dropout = dropout

                self.pred = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x, edge_index):
                for conv, bn in zip(self.convs[:-1], self.bns):
                    x = conv(x, edge_index)
                    x = bn(x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

                x = self.convs[-1](x, edge_index)
                return x

            def predict_edge(self, z, edge_index):
                z_i = z[edge_index[0]]
                z_j = z[edge_index[1]]
                z_concat = torch.cat([z_i, z_j], dim=1)
                return self.pred(z_concat).squeeze(-1)

        self.model = HostVirusGNN(input_dim, self.hidden_dim, self.num_layers, self.dropout)
        return self.model

    def fit(
        self,
        data: "Data",
        epochs: int = 200,
        learning_rate: float = 0.001,
        val_fraction: float = 0.2,
        early_stopping_patience: int = 20,
        verbose_every: int = 20,
    ) -> Dict:
        """
        Train on a subset of edges, early stop on held-out edge loss.

        The split is over edges only - node features and graph structure are
        shared, so this is not a fully inductive evaluation. Returns the
        per-epoch history plus best_epoch/stopped_early.
        """
        if self.model is None:
            self.build_model(data.x.shape[1])

        self.model.to(self.device)
        data = data.to(self.device)

        # build_graph stores each edge twice (fwd+rev), so split on pairs of
        # rows or the same interaction lands in both train and val
        n_edges = data.edge_index.shape[1]
        n_pairs = n_edges // 2
        rng = np.random.default_rng(self.random_state)
        pair_perm = rng.permutation(n_pairs)
        n_val_pairs = max(1, int(n_pairs * val_fraction))
        val_pairs = set(pair_perm[:n_val_pairs].tolist())

        train_mask = torch.tensor(
            [not (i // 2 in val_pairs) for i in range(n_edges)], dtype=torch.bool
        )
        val_mask = ~train_mask

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_auc": []}
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            z = self.model(data.x, data.edge_index)
            edge_pred = self.model.predict_edge(z, data.edge_index)

            train_loss = criterion(edge_pred[train_mask], data.edge_attr.squeeze(-1)[train_mask])
            train_loss.backward()
            optimizer.step()

            # eval mode matters here, BatchNorm
            self.model.eval()
            with torch.no_grad():
                z_eval = self.model(data.x, data.edge_index)
                edge_pred_eval = self.model.predict_edge(z_eval, data.edge_index)
                val_targets = data.edge_attr.squeeze(-1)[val_mask]
                val_preds = edge_pred_eval[val_mask]
                val_loss = criterion(val_preds, val_targets)

                val_pred_binary = (val_preds > 0.5).float()
                val_accuracy = (val_pred_binary == val_targets).float().mean()

                val_auc = self._safe_auc(
                    val_targets.cpu().numpy(), val_preds.cpu().numpy()
                )

            history["train_loss"].append(train_loss.item())
            history["val_loss"].append(val_loss.item())
            history["val_accuracy"].append(val_accuracy.item())
            history["val_auc"].append(val_auc)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % verbose_every == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: train_loss={train_loss.item():.4f} "
                    f"val_loss={val_loss.item():.4f} val_acc={val_accuracy.item():.4f} "
                    f"val_auc={val_auc:.4f}"
                )

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch+1})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        history["best_epoch"] = best_epoch
        history["stopped_early"] = patience_counter >= early_stopping_patience
        self.training_history = history
        self._is_fitted = True

        logger.info(
            f"GNN training complete. Best epoch: {best_epoch+1}, "
            f"best val_loss: {best_val_loss:.4f}"
        )

        return history

    @staticmethod
    def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """nan instead of raising when the batch is single-class."""
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))

    def predict(self, virus_indices: List[int], host_indices: List[int]) -> np.ndarray:
        """Indices are into the graph from build_graph; the two lists pair up positionally."""
        if self.model is None or not self._is_fitted:
            raise ValueError("Model must be trained first (call .fit())")

        self.model.eval()

        with torch.no_grad():
            z = self.model(
                self.node_features.to(self.device), self.edge_index.to(self.device)
            )

            v_idx_t = torch.tensor(virus_indices, dtype=torch.long, device=self.device)
            h_idx_t = torch.tensor(
                [self.n_viruses + h for h in host_indices],
                dtype=torch.long,
                device=self.device,
            )
            edge_index_query = torch.stack([v_idx_t, h_idx_t])
            preds = self.model.predict_edge(z, edge_index_query)

        return preds.cpu().numpy()

    def predict_all_hosts(
        self, virus_idx: int, host_indices: List[int]
    ) -> Dict[str, float]:
        """One virus against many hosts, keyed by host name."""
        predictions = self.predict([virus_idx] * len(host_indices), host_indices)

        results = {}
        for h_idx, pred in zip(host_indices, predictions):
            host_name = (
                self.host_names[h_idx] if h_idx < len(self.host_names) else f"Host_{h_idx}"
            )
            results[host_name] = float(pred)

        return results

    def save(self, path):
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict() if self.model is not None else None,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "input_dim": self.node_features.shape[1]
                if self.node_features is not None
                else None,
                "virus_names": self.virus_names,
                "host_names": self.host_names,
                "n_viruses": self.n_viruses,
                "n_hosts": self.n_hosts,
                "is_fitted": self._is_fitted,
                "training_history": self.training_history,
            },
            path,
        )
        logger.info(f"GNN model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["input_dim"] is not None:
            self.build_model(checkpoint["input_dim"])
            self.model.load_state_dict(checkpoint["state_dict"])
        self.virus_names = checkpoint["virus_names"]
        self.host_names = checkpoint["host_names"]
        self.n_viruses = checkpoint["n_viruses"]
        self.n_hosts = checkpoint["n_hosts"]
        self._is_fitted = checkpoint["is_fitted"]
        self.training_history = checkpoint.get("training_history", {})
        logger.info(f"GNN model loaded from {path}")
        return self
