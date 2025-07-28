import os
import argparse
import pandas as pd
from typing import List


def load_reactome_gene_ids(path: str) -> List[str]:
    """
    Load Reactome gene IDs (without version) from gene_all.txt, preserving order.
    """
    with open(path) as f:
        return [line.strip() for line in f]


def process_and_format_gct_file(
    gct_path: str,
    reactome_genes_ordered: List[str],
    output_dir: str,
    subdir_name: str = None
) -> None:
    """
    Load GTEx .gct TPM file, filter and reorder rows to match reactome_genes_ordered,
    fill missing genes with zeros, and save as RNA_all_TPM.txt in output_dir
    (optionally inside a subdirectory).

    Args:
        gct_path: Path to the input .gct file.
        reactome_genes_ordered: List of Reactome gene IDs in desired row order.
        output_dir: Base output directory.
        subdir_name: Optional subdirectory name for saving the file.
    """
    print(f"Processing {gct_path} ...")
    df = pd.read_csv(gct_path, sep='\t', skiprows=2)
    df = df.rename(columns={'Name': 'gene_id'}).drop(columns=['Description'], errors='ignore')

    # Remove version suffix from gene_id (e.g. ENSG000001234.5 -> ENSG000001234)
    df['gene_id'] = df['gene_id'].str.replace(r'\.\d+', '', regex=True)

    # Set gene_id as index
    df = df.set_index('gene_id')

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # Filter to only genes in reactome_genes_ordered (intersection)
    df_filtered = df.loc[df.index.intersection(reactome_genes_ordered)]

    # Reindex to include all genes in reactome_genes_ordered, missing genes filled with zeros
    df_reindexed = df_filtered.reindex(reactome_genes_ordered, fill_value=0)

    # Optional: if you want to keep gene_name as a column (from original df), you could merge here

    # Prepare output path
    save_path = output_dir
    if subdir_name:
        save_path = os.path.join(output_dir, subdir_name)
    os.makedirs(save_path, exist_ok=True)

    out_file = os.path.join(save_path, 'RNA_all_TPM.txt')

    # Save to file with gene IDs as the first column (index)
    df_reindexed.to_csv(out_file, sep='\t', index=True, index_label='gene_id')
    print(f"Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Process GTEx .gct files directly to RNA_all_TPM.txt per tissue")
    parser.add_argument('--input_dir', type=str, default='gtex_v10_tissue_tpms',
                        help='Directory containing original GTEx .gct TPM files.')
    parser.add_argument('--output_dir', type=str, default='../data',
                        help='Directory to save processed RNA_all_TPM.txt files.')
    parser.add_argument('--gene_list', type=str, default='../data/gene_all.txt',
                        help="Path to Reactome gene list file (gene_all.txt).")

    args = parser.parse_args()

    reactome_genes_ordered = load_reactome_gene_ids(args.gene_list)

    for file_name in os.listdir(args.input_dir):
        if not file_name.endswith('.gct'):
            continue
        tissue_name = os.path.splitext(file_name)[0]
        gct_path = os.path.join(args.input_dir, file_name)
        process_and_format_gct_file(gct_path, reactome_genes_ordered, args.output_dir, subdir_name=tissue_name)


if __name__ == "__main__":
    main()
