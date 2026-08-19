#!/usr/bin/env python3
"""Predict dose-dependent total Cmax for arbitrary molecule/regimen rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def build_model(config: dict, degree, device):
    from models.net import net

    return net(
        degree,
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
        set_layer=config["params"].get("set_layer", "SetRep"),
        dose_mode=True,
        device=device,
    ).to(device)


def predict(model, loader, device) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            reg_pred, _, _, _ = model(
                mol_x=data.mol_x,
                mol_x_feat=data.mol_x_feat,
                bond_x=data.mol_edge_attr,
                atom_edge_index=data.mol_edge_index,
                clique_x=data.clique_x,
                clique_edge_index=data.clique_edge_index,
                atom2clique_index=data.atom2clique_index,
                mol_batch=data.mol_x_batch,
                clique_batch=data.clique_x_batch,
                mol_dose_label=data.mol_dose_label,
                save_cluster=False,
            )
            values.extend(reg_pred.reshape(-1).detach().cpu().numpy().tolist())
    return np.asarray(values, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--checkpoint-files", required=True, help="Comma-separated model.pt files")
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    sys.path.insert(0, str(args.model_root.resolve()))
    from torch_geometric.loader import DataLoader
    from utils import ligand_init
    from utils.dataset_dose import MotifMoleculeDataset_dose

    frame = pd.read_csv(args.input_csv)
    required = {"compound_id", "smiles", "dose_mg_day"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input lacks columns: {sorted(missing)}")
    model_input = frame.copy()
    model_input["dose"] = model_input["dose_mg_day"].astype(float)
    ligand_dict = ligand_init(model_input["smiles"].astype(str).drop_duplicates().tolist())
    dataset = MotifMoleculeDataset_dose(model_input, ligand_dict, device=args.device, cache_transform=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, follow_batch=["mol_x", "clique_x"])
    config = json.loads(args.config_json.read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    columns = []
    checkpoint_files = [Path(p.strip()) for p in args.checkpoint_files.split(",") if p.strip()]
    if not checkpoint_files:
        raise ValueError("No Cmax checkpoints supplied")
    for index, checkpoint in enumerate(checkpoint_files, start=1):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        degree_path = checkpoint.parent / "degree.pt"
        if degree_path.exists():
            payload = torch.load(degree_path, map_location="cpu")
            degree = payload.get("ligand_deg", payload)
        else:
            degree = None
        model = build_model(config, degree, device)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        column = f"predicted_log10_cmax_ug_ml_member_{index}"
        frame[column] = predict(model, loader, device)
        columns.append(column)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    values = frame[columns].to_numpy(float)
    frame["predicted_log10_cmax_ug_ml_mean"] = values.mean(axis=1)
    frame["predicted_log10_cmax_ug_ml_sd"] = values.std(axis=1, ddof=1) if len(columns) > 1 else 0.0
    linear = np.power(10.0, values)
    frame["predicted_cmax_ug_ml_mean"] = linear.mean(axis=1)
    frame["predicted_cmax_ug_ml_sd"] = linear.std(axis=1, ddof=1) if len(columns) > 1 else 0.0
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}: {frame.shape}")


if __name__ == "__main__":
    main()
