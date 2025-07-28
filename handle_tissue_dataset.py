import torch
from typing import Dict

import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import SpectralEmbedding
import numpy as np

from utils import load_tissue_data, load_pathways


class TissueDataHandler:
    def __init__(self, base_dir: str, device: torch.device):
        self.device = device
        self.base_dir = base_dir

        self.gene_pathway, self.pathway_network = load_pathways(base_dir, device)
        self.gene_pathway = self.gene_pathway.bool().to(device)

        # Assert loading outputs shapes and types
        assert self.gene_pathway.ndim == 2, "gene_pathway must be 2D (P, G)"
        assert self.gene_pathway.dtype in [torch.bool, torch.uint8, torch.int64], "gene_pathway should be binary"
        assert self.pathway_network.ndim == 2, "pathway_network must be 2D"
        assert self.pathway_network.shape[0] == self.pathway_network.shape[1], "pathway_network must be square"

        self.tissue_data: Dict[str, Dict[str, torch.Tensor]] = {}

        # After self.pathway_network is loaded
        adj = self.pathway_network.float().cpu().numpy()
        assert adj.shape[0] == adj.shape[1], "Adjacency matrix must be square"

        # Build graph
        G = nx.from_numpy_array(adj)
        deg = np.array([val for _, val in G.degree()])
        clust = np.array([val for _, val in nx.clustering(G).items()])

        # Spectral embedding
        embedding = SpectralEmbedding(n_components=8, affinity='precomputed')
        spec = embedding.fit_transform(adj)  # (P, 8)
        assert spec.shape[1] == 8, "Spectral embedding must have 8 features"

        # Combine: degree, clustering, spectral
        features = np.stack([deg, clust], axis=1)
        combined = np.concatenate([features, spec], axis=1)  # (P, 10)
        assert combined.shape[1] == 10, "Combined features must have 10 dims"

        # Normalize
        combined = StandardScaler().fit_transform(combined)
        self.pathway_features = torch.tensor(combined, dtype=torch.float32, device=device)  # (P, 10)
        assert self.pathway_features.shape[1] == 10, "Pathway features must have 10 dims"

    def get_tissue(self, tissue_enum: str):
        if tissue_enum not in self.tissue_data:
            x = load_tissue_data(self.base_dir, tissue_enum, self.device).squeeze(-1)  # (N, G_selected)
            assert x.ndim == 2, f"x_all should be 2D, got {x.shape}"

            # This mask selects the G_selected (nonzero) columns from self.gene_pathway (shape: P x G_total)
            # gene_pathway_sub = self.gene_pathway[gene_mask, :]
            gene_pathway_sub = self.gene_pathway
            assert gene_pathway_sub.shape[0] == x.shape[1], "Mismatch in active gene count"

            self.tissue_data[tissue_enum] = {
                "x": x,
                "gene_pathway_sub": gene_pathway_sub
            }

        return self.tissue_data[tissue_enum]