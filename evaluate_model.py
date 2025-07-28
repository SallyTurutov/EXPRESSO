import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, recall_score
from tqdm import tqdm

from handle_mol_dataset import load_dataset
from utils import load_pathways, set_seed, load_config
from model import EXPRESSO
from consts import TISSUES


def update_best_auc_if_improved(
    current_auc: dict[str, float],
    best_auc_so_far: dict[str, float] | None
) -> tuple[dict[str, float], bool]:
    """
    Compares current epoch AUCs with best seen so far and determines if total regret improved.

    Args:
        current_auc (dict[str, float]): Tissue -> AUC dict for current epoch.
        best_auc_so_far (dict[str, float] | None): Best AUC seen so far per tissue.

    Returns:
        tuple[dict[str, float], bool]: Updated best AUC dict and a bool flag if this epoch is best so far.
    """
    if best_auc_so_far is None:
        # First epoch — treat it as best
        return current_auc.copy(), True

    updated_best = {
        t: max(current_auc.get(t, 0.0), best_auc_so_far.get(t, 0.0))
        for t in best_auc_so_far
    }

    # Compute regret
    current_regret = sum(
        max(updated_best[t] - current_auc.get(t, 0.0), 0.0)
        for t in updated_best
    )

    best_regret = sum(
        max(updated_best[t] - best_auc_so_far.get(t, 0.0), 0.0)
        for t in updated_best
    )

    is_lowest_regret = current_regret < best_regret
    return updated_best, is_lowest_regret


def compute_metrics(y_true: List[int], y_pred_probs: List[float]) -> Tuple[float, float, float]:
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = float('nan')
    y_pred_bin = [1 if p >= 0.5 else 0 for p in y_pred_probs]
    sensitivity = recall_score(y_true, y_pred_bin, pos_label=1)
    specificity = recall_score(y_true, y_pred_bin, pos_label=0)
    return auc, sensitivity, specificity


@torch.no_grad()
def evaluate_model(
        model: EXPRESSO,
        dataset: dict,
        device: torch.device,
        n_seeds: int = 5,
        is_test: bool = True
) -> dict[TISSUES, float]:
    model.eval()
    description = "TEST" if is_test else "VAL"
    print(f"\n{description}:")

    tissue_auc = {}
    for tissue_enum, loaders in dataset.items():
        loaders = dataset[tissue_enum]
        loader = loaders.get("test" if is_test else "val")
        if loader is None:
            continue

        aucs, sens, specs = [], [], []
        for seed in range(n_seeds):
            set_seed(seed)
            y_true, y_pred = [], []

            for batch in loader:
                smiles = batch["smiles"]
                labels = batch["label"].to(device)

                probs = model(smiles, tissue_enum)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(probs.cpu().numpy())

            auc, sensitivity, specificity = compute_metrics(y_true, y_pred)
            aucs.append(auc)
            sens.append(sensitivity)
            specs.append(specificity)

        def summary(x):
            mean = np.mean(x)
            ci = 1.96 * np.std(x) / np.sqrt(len(x))
            return f"{mean:.4f} ± {ci:.4f}"

        print(f"\n[TISSUE: {tissue_enum.name}]")
        print(f"AUC        : {summary(aucs)}")
        print(f"Sensitivity: {summary(sens)}")
        print(f"Specificity: {summary(specs)}")
        tissue_auc[tissue_enum] = np.mean(aucs)

    return tissue_auc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=Path, default=Path("data/"))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_path', type=Path, required=True)
    parser.add_argument('--n_seeds', type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = load_config(args.save_path.parent)
    dataset = load_dataset(args.base_dir, batch_size=args.batch_size)

    model = EXPRESSO.load(Path(args.save_path))
    evaluate_model(model, dataset, device, args.n_seeds)
