#!/usr/bin/env python3
"""Predict PPB for arbitrary SMILES with retrained MotifAttnNet checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif-root", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--checkpoint-dirs", required=True, help="Comma-separated retraining run directories.")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--target", choices=["activity", "neglogfu", "logitfu"], default="logitfu")
    parser.add_argument("--set-layer", default="SetRep")
    parser.add_argument("--clip", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ligand-cache", default=None)
    args = parser.parse_args()

    motif_root = Path(args.motif_root).resolve()
    sys.path.insert(0, str(motif_root))
    sys.path.insert(0, str(motif_root / "tools"))

    from models.net import net
    from utils import ligand_init
    from utils.dataset import MotifMoleculeDataset
    from utils.utils import DataLoader
    from train_ppb_retraining import activity_from_target, activity_to_fu, activity_to_ppb

    input_df = pd.read_csv(args.input_csv)
    smiles = input_df[args.smiles_col].dropna().astype(str).drop_duplicates().tolist()
    pred_df = pd.DataFrame({"smiles": smiles, "activity": np.zeros(len(smiles), dtype=float)})

    ligand_cache = Path(args.ligand_cache) if args.ligand_cache else Path(args.out_csv).with_suffix(".ligand.pt")
    if ligand_cache.exists():
        ligand_dict = torch.load(str(ligand_cache), map_location="cpu")
    else:
        ligand_dict = ligand_init(smiles)
        torch.save(ligand_dict, str(ligand_cache))

    dataset = MotifMoleculeDataset(pred_df, ligand_dict, device=args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, follow_batch=["mol_x", "clique_x"])

    config_path = Path(args.config_path) if args.config_path else motif_root / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    checkpoint_dirs = [Path(item.strip()) for item in args.checkpoint_dirs.split(",") if item.strip()]
    all_pred_activity = []
    all_pred_ppb = []
    all_pred_fu = []

    for ckpt_dir in checkpoint_dirs:
        degree_path = ckpt_dir / "degree.pt"
        if degree_path.exists():
            degree_dict = torch.load(str(degree_path), map_location="cpu")
            mol_deg, _clique_deg = degree_dict["ligand_deg"], degree_dict.get("clique_deg")
        else:
            mol_deg = None

        model = net(
            mol_deg,
            mol_in_channels=config["params"]["mol_in_channels"],
            hidden_channels=config["params"]["hidden_channels"],
            num_layers=config["params"]["num_layers"],
            clique_num_timesteps=config["params"]["clique_num_timesteps"],
            num_timesteps=config["params"]["num_timesteps"],
            n_hidden_sets=config["params"]["n_hidden_sets"],
            n_elements=config["params"]["n_elements"],
            dropout=config["params"]["dropout"],
            regression_head=True,
            classification_head=False,
            multiclassification_head=0,
            set_layer=args.set_layer,
            dose_mode=False,
            device=torch.device(args.device),
        ).to(args.device)
        model.load_state_dict(torch.load(str(ckpt_dir / "save_model" / "model.pt"), map_location=args.device), strict=False)
        model.eval()

        row_smiles = []
        pred_target = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(args.device)
                reg_pred, _, _, _ = model(
                    mol_x=batch.mol_x,
                    mol_x_feat=batch.mol_x_feat,
                    bond_x=batch.mol_edge_attr,
                    atom_edge_index=batch.mol_edge_index,
                    clique_x=batch.clique_x,
                    clique_edge_index=batch.clique_edge_index,
                    atom2clique_index=batch.atom2clique_index,
                    mol_batch=batch.mol_x_batch,
                    clique_batch=batch.clique_x_batch,
                )
                row_smiles.extend(list(batch.mol_key))
                pred_target.append(reg_pred.detach().cpu().numpy().reshape(-1))

        target_np = np.concatenate(pred_target)
        activity_np = activity_from_target(target_np, args.target, args.clip)
        ppb_np = activity_to_ppb(activity_np)
        fu_np = activity_to_fu(activity_np)
        seed_name = ckpt_dir.name
        out_part = pd.DataFrame(
            {
                "smiles": row_smiles,
                f"{seed_name}_predicted_target": target_np,
                f"{seed_name}_predicted_activity": activity_np,
                f"{seed_name}_pred_ppb": ppb_np,
                f"{seed_name}_pred_fu": fu_np,
            }
        )
        pred_df = pred_df.merge(out_part, on="smiles", how="left")
        all_pred_activity.append(activity_np)
        all_pred_ppb.append(ppb_np)
        all_pred_fu.append(fu_np)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_df["predicted_activity_mean"] = np.vstack(all_pred_activity).mean(axis=0)
    pred_df["pred_ppb_mean"] = np.vstack(all_pred_ppb).mean(axis=0)
    pred_df["pred_fu_mean"] = np.vstack(all_pred_fu).mean(axis=0)
    pred_df["pred_ppb_std"] = np.vstack(all_pred_ppb).std(axis=0)
    pred_df["pred_fu_std"] = np.vstack(all_pred_fu).std(axis=0)
    pred_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}: {pred_df.shape}")


if __name__ == "__main__":
    main()
