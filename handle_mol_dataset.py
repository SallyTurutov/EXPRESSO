import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from rdkit import Chem, RDLogger

from consts import TISSUES, TISSUES_TO_USE, SMILES_DATA_FILE, TISSUE_ENUM_TO_SMILES_LABEL

RDLogger.DisableLog('rdApp.*')


class MoleculeDataset(Dataset):
    """Dataset for molecule property prediction, including tissue and organism."""

    def __init__(self, smiles_list, labels, tissues: list[TISSUES]):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tissues = tissues

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):

        return {
            "smiles": self.smiles_list[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
            "tissue": self.tissues[idx].value,
        }


def load_dataset(
        base_dir: Path,
        batch_size: int = 64,
        num_workers: int = 0,
) -> dict[TISSUES, dict[str, DataLoader]]:
    """
    Load SMILES dataset, organize by tissue, return train/val/test DataLoaders for each valid tissue.
    Also collects valid organisms per tissue if requested.

    Args:
        base_dir (Path): Path to the data directory.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): DataLoader parallelism.

    Returns:
        dict[TISSUES, dict[str, DataLoader]]: Mapping from tissue enum to train/val/test DataLoaders.
    """
    df = pd.read_csv(base_dir / SMILES_DATA_FILE)
    results = {}

    for tissue_enum in TISSUES_TO_USE:
        # Assume `tissue_enum` is a TISSUES enum (e.g. TISSUES.BRAIN_CORTEX)
        smiles_tissue_name = TISSUE_ENUM_TO_SMILES_LABEL[tissue_enum]
        tissue_df = df[df["tissue"] == smiles_tissue_name]

        # Filter invalid SMILES
        valid_mask = tissue_df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        final_df = tissue_df[valid_mask]

        # Prepare tensors by split
        loaders = {}

        for split in ["train", "val", "test"]:
            split_df = final_df[final_df["split"] == split].reset_index(drop=True)
            if split_df.empty:
                continue
            dataset = MoleculeDataset(
                smiles_list=split_df["smiles"].tolist(),
                labels=split_df["label"].tolist(),
                tissues=[tissue_enum] * len(split_df),
                # organisms=org_enums[:len(split_df)]
            )
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers
            )

        results[tissue_enum] = loaders

    return results


def load_train_mixed_dataset(base_dir: Path, batch_size: int = 64) -> list:
    """
    Load the full SMILES dataset with mixed tissues.
    """
    dataset = load_dataset(base_dir, batch_size)

    # Pre-load all batches from all loaders
    all_batches = []
    for tissue_enum, loaders in dataset.items():
        train_loader = loaders["train"]
        for batch in train_loader:
            batch["tissue_enum"] = tissue_enum  # keep track of tissue for model forward
            all_batches.append(batch)

    # Shuffle batches from all tissues
    random.shuffle(all_batches)

    return all_batches
