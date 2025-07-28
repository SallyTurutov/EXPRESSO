import torch
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import EXPRESSO
from consts import TISSUES_TO_USE
from handle_tissue_dataset import TissueDataHandler


# 1. Load pathway names
def load_pathway_names(path='data/pathways.txt'):
    df = pd.read_csv(path, sep='\t')
    names = df.iloc[:, 0].tolist()  # assuming first column is pathway name
    return names


# 2. Load model checkpoint
def load_model(checkpoint_path, device):
    model = EXPRESSO.load(Path(checkpoint_path))
    model.to(device)
    model.eval()
    return model


# 3. Get pathway embeddings or attention scores
def get_pathway_importance(model, pathway_names):
    model.eval()
    pathway_scores = {}

    for tissue in tqdm(TISSUES_TO_USE):
        with torch.no_grad():
            scores = model.tissue_encoder.get_pathway_importance(tissue)

        score_dict = {name: float(score) for name, score in zip(pathway_names, scores)}
        pathway_scores[tissue] = score_dict
    return pathway_scores


# 4. Save top-K important pathway names per tissue
def save_top_pathways(tissue_to_scores, pathway_names, save_path, top_k=15):
    with open(save_path, 'w') as f:
        for tissue, scores in tissue_to_scores.items():
            # Convert dict to array for argsort
            scores_arr = np.array([scores[p] for p in pathway_names])
            top_indices = np.argsort(scores_arr)[::-1][:top_k]
            top_pathways = [pathway_names[i] for i in top_indices]

            f.write(f"Tissue: {tissue}\n")
            for p in top_pathways:
                f.write(f"  - {p}\n")
            f.write("\n")


# 5. Main
def main():
    model_path = 'checkpoints/model_5.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pathway_names = load_pathway_names('data/pathways.txt')
    model = load_model(model_path, device)
    tissue_to_scores = get_pathway_importance(model, pathway_names)
    save_top_pathways(tissue_to_scores, pathway_names, 'top_pathways_per_tissue.txt')


if __name__ == '__main__':
    main()
