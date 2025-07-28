import random
import json
import argparse
from functools import lru_cache
from pathlib import Path
from argparse import Namespace

import torch
import pandas as pd
import numpy as np

from consts import GENE_ALL_FILE, GENE_DATA_DIR, GENE_SELECT_FILE, PATHWAY_GENE_W, \
    PATHWAY_CROSSTALK_NETWORK, TISSUES, TISSUE_TO_DATA_DIR


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(args):
    """
    Save argparse.Namespace to a JSON file in args.save_path/args.json.
    Converts Path objects to str for JSON compatibility.
    """
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / "args.json"

    args_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
    }

    with config_path.open("w") as f:
        json.dump(args_dict, f, indent=2)


@lru_cache
def load_config(dir_path: Path) -> Namespace:
    config_path = dir_path / "args.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected config at {config_path}")
    with config_path.open("r") as f:
        config = json.load(f)
        config['base_dir'] = Path(config['base_dir'])
        return argparse.Namespace(**config)


@lru_cache
def load_selected_genes(base_dir: Path) -> list[int]:
    """
    Load the indices of selected genes from files.

    Args:
        base_dir (Path): Base directory containing the gene files.

    Returns:
        list[int]: list of selected gene indices.
    """
    gene_all_data = pd.read_csv(base_dir / GENE_ALL_FILE, header=None)
    gene_all_data.columns = ['gene_id']
    gene_all_data['index'] = range(len(gene_all_data))
    gene_all_data = gene_all_data.set_index('gene_id')

    gene_select_data = pd.read_csv(base_dir / GENE_SELECT_FILE, header=None)
    return list(gene_all_data.loc[list(gene_select_data[0]), 'index'])


@lru_cache
def load_pathways(base_dir: Path, device: torch.device) -> tuple[torch.LongTensor, torch.Tensor]:
    """
    Load gene-pathway matrix and crosstalk network for pathway encoding.

    Args:
        base_dir (Path): Base directory containing pathway files.
        device (torch.device): Device to load the tensors onto.

    Returns:
        tuple[LongTensor, Tensor]: Gene-pathway indices and repeated pathway network.
    """
    gene_pathway = np.load(file=base_dir / PATHWAY_GENE_W)
    gene_pathway = torch.LongTensor(gene_pathway)

    pathway_network = np.load(file=base_dir / PATHWAY_CROSSTALK_NETWORK)
    pathway_network[np.isnan(pathway_network)] = 0
    pathway_network = torch.tensor(pathway_network, dtype=torch.float32).to(device)
    return gene_pathway, pathway_network


@lru_cache
def load_tissue_data(base_dir: Path, tissue: TISSUES, device: torch.device) -> torch.Tensor:
    """
    Load and return selected gene expression data for a given tissue.

    Args:
        base_dir (Path): Base directory containing the tissue data.
        tissue (TISSUES): The tissue to load data for.
        device (torch.device): Device to load the data onto.

    Returns:
        Tensor: Tensor of shape (N, G, 1) with selected gene expressions.
    """
    tissue_data_dir = TISSUE_TO_DATA_DIR[tissue]
    gene_data_dir = GENE_DATA_DIR
    gene_select_index = load_selected_genes(base_dir)

    data = np.load(file=base_dir / gene_data_dir / tissue_data_dir / 'data_all.npy')  # (N, G_total, 1)
    data = data[:, gene_select_index, :]  # (N, G_selected, 1)

    return torch.tensor(data, dtype=torch.float32).to(device)
