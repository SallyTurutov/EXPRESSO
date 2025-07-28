import os
import pandas as pd
import numpy as np
import argparse
from typing import List


def load_feature_data(gene_file: str) -> List[str]:
    """Load gene list from the gene_file (one gene per line)."""
    return pd.read_csv(gene_file, header=None)[0].tolist()


def load_feature_types(feature_type_file: str) -> List[str]:
    """Load list of feature type names from the file."""
    df = pd.read_csv(feature_type_file, sep='\t', header=None)
    return [x.split('.')[0] for x in df[0].tolist()]


def process_feature_type_data(
        subdir_path: str,
        feature_name: str,
        genes: List[str],
        samples: List[str]
) -> np.ndarray:
    """
    Load and process a single feature type file. Return a 2D numpy array
    of shape (num_genes, num_samples), filling missing genes with zeros.
    """
    path = os.path.join(subdir_path, f"{feature_name}.txt")
    if not os.path.isfile(path):
        print(f"Warning: {path} not found, filling with zeros.")
        return np.zeros((len(genes), len(samples)))

    df = pd.read_csv(path, sep='\t')
    df = df.rename(columns={df.columns[0]: 'gene_id'})
    df = df.drop_duplicates(subset='gene_id').set_index('gene_id')

    # Ensure gene rows exist for all genes (fill zeros if not)
    df = df.reindex(genes, fill_value=0)

    # Only keep required samples (columns)
    df = df.loc[:, samples].fillna(0)

    return df.values


def process_subdir(
        subdir_path: str,
        genes: List[str],
        feature_types: List[str]
) -> None:
    """
    Process all feature types for a tissue (subdir), and save as a single .npy.
    """
    # Use the first feature file to extract sample names
    example_path = os.path.join(subdir_path, f"{feature_types[0]}.txt")
    if not os.path.isfile(example_path):
        print(f"Skipping {subdir_path}, no feature files found.")
        return

    df_example = pd.read_csv(example_path, sep='\t')
    samples = df_example.columns[1:].tolist()

    data_all = np.zeros((len(samples), len(genes), len(feature_types)))

    for i, feature_name in enumerate(feature_types):
        print(f"Processing {feature_name} in {subdir_path}")
        data_all[:, :, i] = process_feature_type_data(subdir_path, feature_name, genes, samples).T

    np.save(os.path.join(subdir_path, 'data_all.npy'), data_all)
    print(f"Saved shape: {data_all.shape} to {subdir_path}/data_all.npy")


def main(args):
    genes = load_feature_data(args.gene_file)
    feature_types = load_feature_types(args.feature_type_file)

    for subdir_name in os.listdir(args.input_dir):
        subdir_path = os.path.join(args.input_dir, subdir_name)
        if os.path.isdir(subdir_path):
            try:
                process_subdir(subdir_path, genes, feature_types)
            except Exception as e:
                print(f"Error processing {subdir_path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process per-tissue gene feature files into unified .npy format.')
    parser.add_argument('--input_dir', type=str, default='../data')
    parser.add_argument('--gene_file', type=str, default='../data/gene_all.txt')
    parser.add_argument('--feature_type_file', type=str, default='../data/feature_type.txt')
    args = parser.parse_args()

    main(args)
