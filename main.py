import argparse
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from evaluate_model import evaluate_model, update_best_auc_if_improved
from handle_mol_dataset import load_dataset, load_train_mixed_dataset
from pretrain_tissue_encoder import pretrain_tissue_encoder
from utils import set_seed, save_config
from model import EXPRESSO


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EXPRESSO(args, device).to(device)
    model.train()

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = Adam(trainable_params, lr=args.lr)

    dataset = load_dataset(args.base_dir, batch_size=args.batch_size)
    train_batches = load_train_mixed_dataset(args.base_dir, batch_size=args.batch_size)
    criterion = nn.BCEWithLogitsLoss()

    save_config(args)
    best_val_auc = None

    for epoch in range(args.epochs):
        set_seed(epoch)

        losses = []
        aux_gene_losses = []
        aux_cls_losses = []
        aux_pathway_losses = []

        random.shuffle(train_batches)
        for batch in tqdm(train_batches, desc=f"Epoch {epoch + 1}"):
            smiles = batch["smiles"]
            tissue_enum = batch["tissue_enum"]
            labels = batch["label"].to(device)

            preds, aux_losses = model(smiles, tissue_enum, return_aux_losses=True)
            loss = criterion(preds, labels)

            # Extract and accumulate aux losses
            gene_loss = aux_losses.get('gene', torch.tensor(0.0, device=device))
            cls_loss = aux_losses.get('cls', torch.tensor(0.0, device=device))
            pathway_loss = aux_losses.get('pathway', torch.tensor(0.0, device=device))

            aux_gene_losses.append(gene_loss.item())
            aux_cls_losses.append(cls_loss.item())
            aux_pathway_losses.append(pathway_loss.item())

            if args.only_activity_loss:
                pass
            elif args.only_gene_loss:
                loss += args.lambda_gene * gene_loss
            elif args.only_cls_loss:
                loss += args.lambda_cls * cls_loss
            elif args.only_pathway_loss:
                loss += args.lambda_pathway * pathway_loss
            else:
                loss += args.lambda_gene * gene_loss
                loss += args.lambda_cls * cls_loss
                loss += args.lambda_pathway * pathway_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            losses.append(loss.item())

        # Final logging
        print(f"[Epoch {epoch + 1}] "
              f"Loss: {sum(losses) / len(losses):.4f}, "
              f"Gene Loss: {sum(aux_gene_losses) / len(aux_gene_losses):.4f}, "
              f"Cls Loss: {sum(aux_cls_losses) / len(aux_cls_losses):.4f}, "
              f"Pathway Loss: {sum(aux_pathway_losses) / len(aux_pathway_losses):.4f}, ")

        set_seed(args.seed)
        val_auc = evaluate_model(model, dataset, device, 1, is_test=False)
        best_val_auc, is_best_val_epoch = update_best_auc_if_improved(val_auc, best_val_auc)

        if args.save_path and is_best_val_epoch:
            out_path = args.save_path / f"best_model.pt"
            model.save(out_path)
            print(f"Saved model to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--base_dir', type=Path, default=Path("data/"))
    parser.add_argument('--save_path', type=Path, default=Path("checkpoints/"))

    # Training
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)

    # Pre-training
    parser.add_argument('--pretraining_batch_size', type=int, default=1)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--pretrained_tissue_path', type=Path,
                        default=Path("checkpoints/pretrained_tissue_encoder.pt"))

    # Loss
    parser.add_argument('--lambda_gene', type=float, default=0.5)
    parser.add_argument('--lambda_cls', type=float, default=0.5)
    parser.add_argument('--lambda_pathway', type=float, default=0.25)

    # Molecule Encoder
    parser.add_argument('--mol_proj_hidden_dim', type=int, default=512)
    parser.add_argument('--mol_proj_output_dim', type=int, default=512)

    # Tissue Encoder
    parser.add_argument('--tissue_heads', type=int, default=3)
    parser.add_argument('--tissue_layers', type=int, default=2)
    parser.add_argument('--tissue_dropout', type=int, default=0.1)
    parser.add_argument('--tissue_hidden_dim', type=int, default=192)
    parser.add_argument('--tissue_output_dim', type=int, default=192)

    parser.add_argument('--tissue_proj_hidden_dim', type=int, default=256)
    parser.add_argument('--tissue_proj_output_dim', type=int, default=256)

    # Activity Classifier
    parser.add_argument('--classifier_hidden_dim', type=int, default=512)
    parser.add_argument('--classifier_dropout', type=int, default=0.2)

    # Modality Dropout
    parser.add_argument('--no_modality_dropout', action='store_true')
    parser.add_argument('--modality_dropout_ratio', type=int, default=0.25)

    # Ablation
    parser.add_argument('--no_pretraining', action='store_true')
    parser.add_argument('--no_tissue_embedding', action='store_true')
    parser.add_argument('--only_activity_loss', action='store_true')
    parser.add_argument('--only_gene_loss', action='store_true')
    parser.add_argument('--only_cls_loss', action='store_true')
    parser.add_argument('--only_pathway_loss', action='store_true')
    parser.add_argument('--only_sample_nodes', action='store_true')
    parser.add_argument('--no_gene_nodes', action='store_true')
    parser.add_argument('--no_pathway_nodes', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"{args=}")

    set_seed(args.seed)
    args.save_path.mkdir(parents=True, exist_ok=True)

    if args.no_pretraining or args.only_activity_loss or args.no_tissue_embedding:
        args.pretrained_tissue_path = None
    else:
        pretrain_tissue_encoder(args)

    train_model(args)
