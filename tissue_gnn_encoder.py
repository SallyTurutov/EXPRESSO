import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv

from consts import TISSUES_PRETRAIN_ALL
from handle_tissue_dataset import TissueDataHandler


class MultiQueryPooling(nn.Module):
    def __init__(self, dim_input, dim_hidden=192, dim_output=192, num_heads=4, num_queries=20, ln=True):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(1, num_queries, dim_input))  # (1, Q, D)
        self.mab = nn.MultiheadAttention(dim_input, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_output)
        )
        self.ln = nn.LayerNorm(dim_output) if ln else nn.Identity()

    def forward(self, x):
        x = x.unsqueeze(0)  # (1, N, D)
        q = self.query.expand(x.size(0), -1, -1)  # (1, Q, D)
        attn_out, _ = self.mab(q, x, x)  # (1, Q, D)
        pooled = attn_out.mean(dim=1)  # (1, D)
        return self.ln(self.ff(pooled))  # (1, D)


class HeteroGNNBlock(nn.Module):
    def __init__(self, hidden_dim, heads: int, dropout: int, args):
        super().__init__()
        out_channels = hidden_dim // heads

        if args.only_sample_nodes:
            self.conv = HeteroConv({
                ('sample', 'expresses', 'gene'): SAGEConv((-1, -1), hidden_dim),
                ('gene', 'expressed_by', 'sample'): SAGEConv((-1, -1), hidden_dim),
            }, aggr='sum')
            self.norms = nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim) for node_type in ['sample', 'gene']
            })
        elif args.no_gene_nodes:
            self.conv = HeteroConv({
                ('sample', 'expresses', 'gene'): SAGEConv((-1, -1), hidden_dim),
                ('gene', 'expressed_by', 'sample'): SAGEConv((-1, -1), hidden_dim),
                ('pathway', 'pathway_crosstalk', 'pathway'): GATv2Conv((-1, -1), out_channels, heads=heads,
                                                                       add_self_loops=False),
            }, aggr='sum')
            self.norms = nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim) for node_type in ['sample', 'gene', 'pathway']
            })
        elif args.no_pathway_nodes:
            self.conv = HeteroConv({
                ('sample', 'expresses', 'gene'): SAGEConv((-1, -1), hidden_dim),
                ('gene', 'expressed_by', 'sample'): SAGEConv((-1, -1), hidden_dim),
                ('gene', 'in_pathway', 'pathway'): GATv2Conv((-1, -1), out_channels, heads=heads, add_self_loops=False),
                ('pathway', 'has_gene', 'gene'): GATv2Conv((-1, -1), out_channels, heads=heads, add_self_loops=False),
            }, aggr='sum')
            self.norms = nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim) for node_type in ['sample', 'gene', 'pathway']
            })

        else:
            self.conv = HeteroConv({
                ('sample', 'expresses', 'gene'): SAGEConv((-1, -1), hidden_dim),
                ('gene', 'expressed_by', 'sample'): SAGEConv((-1, -1), hidden_dim),
                ('gene', 'in_pathway', 'pathway'): GATv2Conv((-1, -1), out_channels, heads=heads, add_self_loops=False),
                ('pathway', 'has_gene', 'gene'): GATv2Conv((-1, -1), out_channels, heads=heads, add_self_loops=False),
                ('pathway', 'pathway_crosstalk', 'pathway'): GATv2Conv((-1, -1), out_channels, heads=heads,
                                                                       add_self_loops=False),
            }, aggr='sum')
            self.norms = nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim) for node_type in ['sample', 'gene', 'pathway']
            })
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        out = self.conv(x_dict, edge_index_dict)
        out_res = {}

        for node_type in x_dict:
            h_in = x_dict[node_type]
            h_out = out[node_type]

            assert h_out.shape == h_in.shape, f"Output shape {h_out.shape} and input shape {h_in.shape} " \
                                              f"mismatch for node {node_type}"

            if h_in.shape == h_out.shape:
                h_out = h_out + h_in  # Residual connection

            h_out = self.norms[node_type](h_out)
            h_out = self.dropout(h_out)

            out_res[node_type] = h_out

        return out_res


class TissueGNNEncoder(nn.Module):
    def __init__(self, base_dir: str, heads: int, layers: int, dropout: int, hidden_dim: int,
                 output_dim: int, device: torch.device, args):
        super().__init__()
        self.device = device
        self.data_handler = TissueDataHandler(base_dir, device)

        self.gnn_layers = nn.ModuleList([
            HeteroGNNBlock(hidden_dim, heads=heads, dropout=dropout, args=args) for _ in range(layers)
        ])

        self.num_total_genes = self.data_handler.gene_pathway.shape[1]
        self.gene_expr_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gene_proj = nn.Linear(11560, hidden_dim).to(self.device)

        self.pathway_proj = nn.Linear(10, hidden_dim)
        self.pool = MultiQueryPooling(dim_input=hidden_dim, dim_output=hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self.tissue_classifier = nn.Linear(output_dim, len(TISSUES_PRETRAIN_ALL))
        self.ce_loss = nn.CrossEntropyLoss()

        self.gene_decoder = nn.Linear(hidden_dim, 1)
        self.recon_loss = nn.MSELoss()
        self.pathway_proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Ablation
        self.only_sample_nodes = args.only_sample_nodes
        self.no_gene_nodes = args.no_gene_nodes
        self.no_pathway_nodes = args.no_pathway_nodes

    def build_graph(self, tissue_enum: str, apply_gene_masking=False, mask_ratio=0.15,
                    augment_pathways=False, apply_dropout=True, dropout_rate=0.15, apply_noise=True):
        """
        Build the heterogeneous graph for a tissue.

        Args:
            tissue_enum (str): Tissue name or enum key
            apply_gene_masking (bool): Whether to mask genes for self-supervision
            mask_ratio (float): Fraction of genes to mask

        Returns:
            x_dict, edge_index_dict, mask_info (if masking)
        """
        # Load tissue data: expression matrix (samples × genes) and gene-pathway mask
        data = self.data_handler.get_tissue(tissue_enum)
        x_expr = data['x']  # shape: (N_samples, G_active)
        gene_pathway = data['gene_pathway_sub']  # shape: (G_active, P)

        # Assert shapes and types
        assert x_expr.ndim == 2, f"x_expr must be 2D, got {x_expr.shape}"
        assert gene_pathway.ndim == 2, f"gene_pathway must be 2D, got {gene_pathway.shape}"
        assert x_expr.shape[1] == gene_pathway.shape[
            0], f"Number of genes mismatch between x_expr {x_expr.shape[1]} and gene_pathway {gene_pathway.shape[0]}"

        # Normalize gene expression: log-transform and standardize
        x_expr = torch.log1p(x_expr)
        x_expr = (x_expr - x_expr.mean()) / (x_expr.std() + 1e-6)

        N, G = x_expr.shape
        P = gene_pathway.shape[1]

        # === NODE FEATURES ===
        # x_expr: (N, G_active)
        # Step 1: Project each scalar gene expression to embedding
        gene_input = x_expr.unsqueeze(-1)  # (N, G_active, 1)
        gene_embeds = self.gene_expr_encoder(gene_input)  # (N, G_active, hidden_dim)

        # Step 2: Aggregate per-sample gene embeddings
        sample_feat = gene_embeds.sum(dim=1)  # (N, hidden_dim)

        # Mean expression per gene across samples
        gene_expr = x_expr.mean(dim=0)  # (G_active,)
        gene_feat = self.gene_expr_encoder(gene_expr.unsqueeze(-1))  # (G_active, hidden_dim)

        # Pathway features: only use structural graph embedding (not gene expression aggregation)
        pathway_graph_feat = self.data_handler.pathway_features[:P]  # (P, 10)
        assert pathway_graph_feat.shape[1] == 10, f"Pathway features dim must be 10, got {pathway_graph_feat.shape[1]}"

        # Project features
        if self.only_sample_nodes:
            x_dict = {
                'sample': sample_feat,  # (N, G)
                'gene': gene_feat,  # (G, D)
            }
        else:
            x_dict = {
                'sample': sample_feat,  # (N, G)
                'gene': gene_feat,  # (G, D)
                'pathway': self.pathway_proj(pathway_graph_feat)  # (P, D)
            }

        # === EDGE INDICES ===
        # sample ↔ gene edges where expression is nonzero
        sample_idx, gene_idx = x_expr.nonzero(as_tuple=True)
        edge_index_sg = torch.stack([sample_idx, gene_idx], dim=0)  # shape (2, E_sg)
        assert edge_index_sg.shape[0] == 2, "Edge index sg must have shape (2, E)"

        # gene ↔ pathway edges from gene-pathway membership mask
        gene_idx2, path_idx = gene_pathway.nonzero(as_tuple=True)
        edge_index_gp = torch.stack([gene_idx2, path_idx], dim=0)
        assert edge_index_gp.shape[0] == 2, "Edge index gp must have shape (2, E)"

        # pathway ↔ pathway edges from pathway crosstalk adjacency
        edge_index_pp = self.data_handler.pathway_network.nonzero(as_tuple=False).T.contiguous()
        assert edge_index_pp.shape[0] == 2, "Edge index pp must have shape (2, E)"

        if self.only_sample_nodes:
            edge_index_dict = {
                ('sample', 'expresses', 'gene'): edge_index_sg,
                ('gene', 'expressed_by', 'sample'): edge_index_sg.flip(0),
            }
        elif self.no_gene_nodes:
            edge_index_dict = {
                ('sample', 'expresses', 'gene'): edge_index_sg,
                ('gene', 'expressed_by', 'sample'): edge_index_sg.flip(0),
                ('pathway', 'pathway_crosstalk', 'pathway'): edge_index_pp
            }
        elif self.no_pathway_nodes:
            edge_index_dict = {
                ('sample', 'expresses', 'gene'): edge_index_sg,
                ('gene', 'expressed_by', 'sample'): edge_index_sg.flip(0),
                ('gene', 'in_pathway', 'pathway'): edge_index_gp,
                ('pathway', 'has_gene', 'gene'): edge_index_gp.flip(0),
            }
        else:
            edge_index_dict = {
                ('sample', 'expresses', 'gene'): edge_index_sg,
                ('gene', 'expressed_by', 'sample'): edge_index_sg.flip(0),
                ('gene', 'in_pathway', 'pathway'): edge_index_gp,
                ('pathway', 'has_gene', 'gene'): edge_index_gp.flip(0),
                ('pathway', 'pathway_crosstalk', 'pathway'): edge_index_pp
            }

        # Optional: Gene masking for self-supervised learning
        if apply_gene_masking:
            mask_matrix = (torch.rand_like(x_expr) < mask_ratio)  # (N, G)
            target_values = x_expr[mask_matrix]  # 1D tensor of masked expression values
            x_expr = x_expr.clone()
            x_expr[mask_matrix] = 0.0
            mask_info = (mask_matrix, target_values)
            return x_dict, edge_index_dict, mask_info

        if augment_pathways:
            pathway_feat = self.data_handler.pathway_features[:P].clone()

            if apply_dropout:
                keep_mask = torch.rand(P) > dropout_rate
                pathway_feat[~keep_mask] = 0.0

            if apply_noise:
                noise = torch.randn_like(pathway_feat) * 0.1
                pathway_feat = pathway_feat + noise

            x_dict['pathway'] = self.pathway_proj(pathway_feat)

        return x_dict, edge_index_dict

    def compute_aux_losses(self, tissue_enum: str):
        """
        Compute auxiliary self-supervised losses:
            - gene reconstruction
            - tissue classification
            - pathway contrastive loss (on augmented views)

        Returns:
            dict: {
                'gene': ...,
                'cls': ...,
                'pathway': ...
            }
        """
        # === Graph View 1 ===
        x_dict_1, edge_index_dict_1, mask_info_1 = self.build_graph(
            tissue_enum, apply_gene_masking=True, augment_pathways=True
        )
        for layer in self.gnn_layers:
            x_dict_1 = layer(x_dict_1, edge_index_dict_1)

        # === Graph View 2 ===
        x_dict_2, edge_index_dict_2, _ = self.build_graph(
            tissue_enum, apply_gene_masking=True, augment_pathways=True
        )
        for layer in self.gnn_layers:
            x_dict_2 = layer(x_dict_2, edge_index_dict_2)

        losses = {}

        # === Gene Reconstruction Loss ===
        if mask_info_1 is not None:
            mask_matrix, target_values = mask_info_1
            sample_embed = x_dict_1['sample']  # (N, D)
            gene_embed = x_dict_1['gene']  # (G, D)
            pred_matrix = torch.matmul(sample_embed, gene_embed.T)  # (N, G)
            pred_masked = pred_matrix[mask_matrix]  # (num_masked,)
            losses['gene'] = self.recon_loss(pred_masked, target_values)

        # === Tissue Classification Loss ===
        target_index = TISSUES_PRETRAIN_ALL.index(tissue_enum)
        target = torch.tensor([target_index], device=self.device)
        pooled = self.pool(x_dict_1['sample'])
        tissue_embed = self.out_proj(pooled)
        logits = self.tissue_classifier(tissue_embed)
        losses['cls'] = self.ce_loss(logits, target)

        # === Pathway Contrastive Loss ===
        if 'pathway' in x_dict_1 and 'pathway' in x_dict_2:
            z1 = self.pathway_proj_head(x_dict_1['pathway'])  # (P, D)
            z2 = self.pathway_proj_head(x_dict_2['pathway'])  # (P, D)

            P = z1.size(0)
            sim_pos = F.cosine_similarity(z1, z2, dim=1)
            sim_neg = torch.matmul(z2, z1.T)
            sim_neg.fill_diagonal_(-1e9)  # mask out self

            logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1) / 0.1
            labels = torch.zeros(P, dtype=torch.long, device=z1.device)
            losses['pathway'] = F.cross_entropy(logits, labels)
            losses['pathway_z1'] = z1
            losses['pathway_z2'] = z2

        return losses

    def forward(self, tissue_enum):
        # Build graph features and edges
        x_dict, edge_index_dict = self.build_graph(tissue_enum)

        # Apply stacked heterogeneous GNN blocks
        for gnn_layer in self.gnn_layers:
            x_dict = gnn_layer(x_dict, edge_index_dict)

        # Extract sample node embeddings after GNN
        sample_embeds = x_dict['sample']  # shape: (N_samples, hidden_dim)
        assert sample_embeds.ndim == 2, f"Sample embeddings should be 2D, got {sample_embeds.shape}"

        # Aggregate sample embeddings to get a single tissue embedding vector
        tissue_embedding = self.pool(sample_embeds)  # (1, hidden_dim)
        tissue_embedding = self.out_proj(tissue_embedding)  # project to output dim
        assert tissue_embedding.ndim == 2 and tissue_embedding.shape[
            0] == 1, f"Tissue embedding shape unexpected: {tissue_embedding.shape}"

        return tissue_embedding

    def get_pathway_importance(self, tissue_enum):
        self.eval()
        with torch.no_grad():
            x_dict, edge_index_dict = self.build_graph(tissue_enum)
            for gnn in self.gnn_layers:
                x_dict = gnn(x_dict, edge_index_dict)

            # get the final pathway node embeddings
            pathway_embeddings = x_dict['pathway']  # (P, D)

            # compute some measure of importance, e.g., L2 norm
            importance = pathway_embeddings.norm(dim=1)  # (P,)

            return importance
