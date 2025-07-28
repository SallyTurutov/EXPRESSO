import torch
import torch.nn.functional as F
import random
from tqdm import tqdm

from tissue_gnn_encoder import TissueGNNEncoder
from consts import TISSUES_PRETRAIN_ALL


def pathway_contrastive_loss(z1_list, z2_list, temperature=0.1):
    loss_total = 0.0
    count = 0
    for z1, z2 in zip(z1_list, z2_list):  # each z1/z2 is (P, D)
        P = z1.size(0)
        for i in range(P):
            pos = F.cosine_similarity(z1[i], z2[i], dim=0)  # scalar
            neg = torch.cat([z1[:i], z1[i + 1:]], dim=0)  # (P-1, D)
            neg_sim = F.cosine_similarity(neg, z2[i].expand_as(neg), dim=1)  # (P-1,)

            numerator = torch.exp(pos / temperature)
            denominator = numerator + torch.sum(torch.exp(neg_sim / temperature))
            loss_i = -torch.log(numerator / (denominator + 1e-8))
            loss_total += loss_i
            count += 1
    return loss_total / count if count > 0 else torch.tensor(0.0, device=z1.device)


def pretrain_tissue_encoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TissueGNNEncoder(
        base_dir=args.base_dir,
        layers=args.tissue_layers,
        dropout=args.tissue_dropout,
        heads=args.tissue_heads,
        hidden_dim=args.tissue_hidden_dim,
        output_dim=args.tissue_output_dim,
        device=device,
        args=args
    ).to(device)
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    tissues = list(TISSUES_PRETRAIN_ALL)
    random.shuffle(tissues)
    batch_size = min(args.pretraining_batch_size, len(tissues))

    for epoch in range(args.pretrain_epochs):
        if epoch == 0:
            torch.cuda.synchronize()
        losses = []

        for i in tqdm(range(0, len(tissues), batch_size), desc=f"Epoch {epoch + 1}"):
            batch = tissues[i:i + batch_size]
            gene_loss_total = 0.0
            cls_loss_total = 0.0
            pathway_z1_all, pathway_z2_all = [], []

            for tissue in batch:
                loss_dict = encoder.compute_aux_losses(tissue)
                gene_loss_total += loss_dict.get("gene", 0.0)
                cls_loss_total += loss_dict.get("cls", 0.0)
                if "pathway_z1" in loss_dict and "pathway_z2" in loss_dict:
                    pathway_z1_all.append(loss_dict["pathway_z1"])
                    pathway_z2_all.append(loss_dict["pathway_z2"])

            avg_gene_loss = gene_loss_total / len(batch)
            avg_cls_loss = cls_loss_total / len(batch) if cls_loss_total > 0 else 0.0

            if pathway_z1_all and pathway_z2_all:
                pathway_loss = pathway_contrastive_loss(pathway_z1_all, pathway_z2_all, temperature=0.1)
            else:
                pathway_loss = torch.tensor(0.0, device=device)

            if args.only_gene_loss:
                total_loss = args.lambda_gene * avg_gene_loss
            elif args.only_cls_loss:
                total_loss = args.lambda_cls * avg_cls_loss
            elif args.only_pathway_loss:
                total_loss = args.lambda_pathway * pathway_loss
            else:
                total_loss = args.lambda_gene * avg_gene_loss + args.lambda_cls * avg_cls_loss \
                             + args.lambda_pathway * pathway_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        if epoch == 0:
            torch.cuda.synchronize()
        epoch_loss = sum(losses) / len(losses)
        print(
            f"Epoch {epoch + 1}: Total Loss = {epoch_loss:.4f} | Gene = {avg_gene_loss:.4f} | "
            f"Cls = {avg_cls_loss:.4f} | Pathway = {pathway_loss.item():.4f}"
        )

    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'args': vars(args)
    }, args.pretrained_tissue_path)
    print(f"Pretrained tissue encoder saved to {args.pretrained_tissue_path}")
