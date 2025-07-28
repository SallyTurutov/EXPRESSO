import torch
import torch.nn as nn
from pathlib import Path

from clamp import CLAMP
from utils import load_config
from activity_classifier import ActivityClassifier
from tissue_gnn_encoder import TissueGNNEncoder


class EXPRESSO(nn.Module):
    """
    EXPRESSO full model.
    """

    def __init__(
            self,
            args,
            device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.use_tissue = not args.no_tissue_embedding
        self.use_modality_dropout = not args.no_modality_dropout
        self.modality_dropout_ratio = args.modality_dropout_ratio

        # 1. Clamp model
        self.clamp = CLAMP().to(device)
        self.clamp.eval()  # Keep frozen

        self.mol_dim = 768  # fixed by CLAMP
        self.tissue_dim = args.tissue_output_dim if self.use_tissue else 0

        # 2. Tissue encoder
        if self.use_tissue:
            self.tissue_encoder = TissueGNNEncoder(
                base_dir=args.base_dir,
                layers=args.tissue_layers,
                dropout=args.tissue_dropout,
                heads=args.tissue_heads,
                hidden_dim=args.tissue_hidden_dim,
                output_dim=args.tissue_output_dim,
                device=device,
                args=args
            ).to(device)

            if args.pretrained_tissue_path is not None:
                checkpoint = torch.load(args.pretrained_tissue_path, map_location=device)
                self.tissue_encoder.load_state_dict(checkpoint['encoder_state_dict'])

            # Allow fine-tuning of the tissue encoder
            for param in self.tissue_encoder.parameters():
                param.requires_grad = True

        else:
            self.tissue_encoder = None

        # 3. Classifier
        self.fused_dim = args.mol_proj_output_dim
        self.mol_proj = nn.Sequential(
            nn.Linear(self.mol_dim, args.mol_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.mol_proj_hidden_dim, args.mol_proj_output_dim)
        )

        if self.use_tissue:
            self.tissue_proj = nn.Sequential(
                nn.Linear(self.tissue_dim, args.tissue_proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.tissue_proj_hidden_dim, args.tissue_proj_output_dim)
            )
            self.fused_dim += args.tissue_proj_output_dim

        self.classifier = ActivityClassifier(
            input_dim=self.fused_dim,
            hidden_dim=args.classifier_hidden_dim,
            dropout=args.classifier_dropout
        ).to(device)

    def forward(self, smiles, tissue_enum, return_aux_losses=False):
        """
        Args:
            smiles: List of smiles
            tissue_enum: Enum type
            return_aux_losses (bool): Whether to return aux losses
        Returns:
            logits: [B] unnormalized logit predictions
        """
        with torch.no_grad():
            smiles_emb = self.clamp.compound_encoder(smiles).to(self.device)

        if self.use_tissue:
            tissue_emb = self.tissue_encoder(tissue_enum)
            tissue_emb = tissue_emb.repeat(len(smiles), 1)

            # === Modality Dropout ===
            if self.training and self.use_modality_dropout:
                if torch.rand(1).item() < self.modality_dropout_ratio:
                    smiles_emb = torch.zeros_like(smiles_emb)
                if torch.rand(1).item() < self.modality_dropout_ratio:
                    tissue_emb = torch.zeros_like(tissue_emb)

            mol_emb_proj = self.mol_proj(smiles_emb)
            tissue_emb_proj = self.tissue_proj(tissue_emb)
            x = torch.cat([mol_emb_proj, tissue_emb_proj], dim=-1)
        else:
            x = self.mol_proj(smiles_emb)

        logits = self.classifier(x)

        if return_aux_losses:
            aux_losses = self.tissue_encoder.compute_aux_losses(tissue_enum) if self.use_tissue else {}
            return logits, aux_losses

        return logits

    def save(self, path: str | Path):
        torch.save(self.state_dict(), str(path))

    @classmethod
    def load(cls, checkpoint_path: Path, device: torch.device = None):
        """
        Load EXPRESSO from a checkpoint and its associated config.
        Args:
            checkpoint_path (Path): Path to the saved model weights (.pt file)
            device (torch.device): Device to load the model on
        Returns:
            EXPRESSO instance with weights loaded
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        args = load_config(checkpoint_path.parent)

        # Build model and load weights
        model = cls(args=args, device=device)
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
        return model
