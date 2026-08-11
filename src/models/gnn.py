"""
Graph Neural Network for host-virus interaction prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, DataLoader
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GNNModel:
    """
    Graph Neural Network for predicting host-virus interactions.

    Builds a bipartite graph where:
    - Nodes: Viruses and hosts
    - Edges: Known infection events
    - Features: Node features (genomic, ecological)
    - Target: Edge prediction (susceptibility)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        device: str = "cpu",
    ):
        """
        Initialize GNN model.

        Args:
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout rate
            device: 'cpu' or 'cuda'
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.model = None
        self.node_features = None
        self.edge_index = None

    def build_graph(
        self,
        virus_features: np.ndarray,
        host_features: np.ndarray,
        interactions: List[Tuple[int, int, float]],
        virus_names: List[str],
        host_names: List[str],
    ) -> Data:
        """
        Build graph from features and interactions.

        Args:
            virus_features: Features for virus nodes
            host_features: Features for host nodes
            interactions: List of (virus_idx, host_idx, label)
            virus_names: List of virus names
            host_names: List of host names

        Returns:
            Data: PyTorch Geometric data object
        """
        n_viruses = len(virus_features)
        n_hosts = len(host_features)
        n_nodes = n_viruses + n_hosts

        # Combine features
        features = np.vstack([virus_features, host_features])
        features = torch.tensor(features, dtype=torch.float)

        # Create edge index
        edges = []
        edge_labels = []

        for v_idx, h_idx, label in interactions:
            # Virus to host (forward)
            edges.append([v_idx, n_viruses + h_idx])
            edge_labels.append(label)

            # Host to virus (reverse) - for undirected graph
            edges.append([n_viruses + h_idx, v_idx])
            edge_labels.append(label)

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_labels = torch.tensor(edge_labels, dtype=torch.float)

        # Create PyG data object
        data = Data(
            x=features, edge_index=edge_index, edge_attr=edge_labels.unsqueeze(1)
        )

        self.node_features = features
        self.edge_index = edge_index

        # Store metadata
        self.virus_names = virus_names
        self.host_names = host_names
        self.n_viruses = n_viruses
        self.n_hosts = n_hosts

        return data

    def build_model(self, input_dim: int) -> nn.Module:
        """
        Build the GNN architecture.

        Args:
            input_dim: Input feature dimension

        Returns:
            nn.Module: PyTorch model
        """

        class HostVirusGNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                super().__init__()

                self.convs = nn.ModuleList()
                self.bns = nn.ModuleList()

                # Input layer
                self.convs.append(GCNConv(input_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

                # Hidden layers
                for _ in range(num_layers - 2):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                    self.bns.append(nn.BatchNorm1d(hidden_dim))

                # Output layer
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

                self.dropout = dropout

                # Prediction layer
                self.pred = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x, edge_index):
                # Node embeddings
                for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
                    x = conv(x, edge_index)
                    x = bn(x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

                x = self.convs[-1](x, edge_index)

                return x

            def predict_edge(self, z, edge_index):
                # Predict edge labels
                z_i = z[edge_index[0]]
                z_j = z[edge_index[1]]
                z_concat = torch.cat([z_i, z_j], dim=1)
                return self.pred(z_concat).squeeze()

        self.model = HostVirusGNN(
            input_dim, self.hidden_dim, self.num_layers, self.dropout
        )

        return self.model

    def train(
        self, data: Data, epochs: int = 100, learning_rate: float = 0.001
    ) -> Dict:
        """
        Train the GNN model.

        Args:
            data: PyG data object
            epochs: Number of epochs
            learning_rate: Learning rate

        Returns:
            Dict: Training history
        """
        if self.model is None:
            self.build_model(data.x.shape[1])

        self.model.to(self.device)
        data = data.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            # Forward pass
            z = self.model(data.x, data.edge_index)

            # Edge prediction
            edge_pred = self.model.predict_edge(z, data.edge_index)

            # Loss
            loss = criterion(edge_pred, data.edge_attr.squeeze())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            pred_binary = (edge_pred > 0.5).float()
            accuracy = (pred_binary == data.edge_attr.squeeze()).float().mean()

            history["loss"].append(loss.item())
            history["accuracy"].append(accuracy.item())

            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}, "
                    f"Accuracy = {accuracy.item():.4f}"
                )

        return history

    def predict(self, virus_indices: List[int], host_indices: List[int]) -> np.ndarray:
        """
        Predict susceptibility for virus-host pairs.

        Args:
            virus_indices: List of virus node indices
            host_indices: List of host node indices

        Returns:
            np.ndarray: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        self.model.eval()

        with torch.no_grad():
            z = self.model(
                self.node_features.to(self.device), self.edge_index.to(self.device)
            )

            predictions = []

            for v_idx, h_idx in zip(virus_indices, host_indices):
                z_v = z[v_idx]
                z_h = z[h_idx]

                z_concat = torch.cat([z_v, z_h]).unsqueeze(0)
                pred = self.model.pred(z_concat)
                predictions.append(pred.item())

        return np.array(predictions)

    def predict_all_hosts(
        self, virus_idx: int, host_indices: List[int]
    ) -> Dict[str, float]:
        """
        Predict susceptibility for one virus against multiple hosts.

        Args:
            virus_idx: Virus node index
            host_indices: List of host node indices

        Returns:
            Dict[str, float]: Predictions per host
        """
        predictions = self.predict([virus_idx] * len(host_indices), host_indices)

        results = {}
        for h_idx, pred in zip(host_indices, predictions):
            host_name = (
                self.host_names[h_idx]
                if h_idx < len(self.host_names)
                else f"Host_{h_idx}"
            )
            results[host_name] = pred

        return results
