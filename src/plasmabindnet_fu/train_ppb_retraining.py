#!/usr/bin/env python3
"""Retrain MotifAttnNet PPB with downstream-aware PPB/fu evaluation.

This script is intended to run inside the remote MotifAttnNet project root.
It reuses the existing model, dataset, ligand graph cache, and split CSVs,
but makes the target transform, regression loss, bin weighting, calibration,
and reporting explicit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.net import net
from utils import ligand_init
from utils.dataset import MotifMoleculeDataset
from utils.utils import CustomWeightedRandomSampler, DataLoader, compute_pna_degrees


PPB_BINS = np.array([0.0, 0.4, 0.8, 0.9, 0.95, 0.99, 1.000001], dtype=float)
PPB_BIN_LABELS = ["0-40", "40-80", "80-90", "90-95", "95-99", "99-100"]


def same_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def activity_to_fu(activity: np.ndarray | torch.Tensor, eps: float = 1e-12):
    if isinstance(activity, torch.Tensor):
        return torch.clamp(torch.pow(torch.tensor(10.0, device=activity.device), activity.float()), eps, 1.0)
    return np.clip(np.power(10.0, np.asarray(activity, dtype=float)), eps, 1.0)


def activity_to_ppb(activity: np.ndarray | torch.Tensor):
    if isinstance(activity, torch.Tensor):
        return torch.clamp((1.0 - activity_to_fu(activity)) / 0.999, 0.0, 1.0)
    return np.clip((1.0 - activity_to_fu(activity)) / 0.999, 0.0, 1.0)


def target_from_activity(activity: torch.Tensor, target: str, clip: float) -> torch.Tensor:
    fu = activity_to_fu(activity, eps=clip)
    fu = torch.clamp(fu, clip, 1.0 - clip)
    if target == "activity":
        return activity.float()
    if target == "neglogfu":
        return -torch.log10(fu)
    if target == "logitfu":
        return torch.log(fu / (1.0 - fu))
    raise ValueError(f"Unknown target: {target}")


def activity_from_target(pred: np.ndarray, target: str, clip: float) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if target == "activity":
        return pred
    if target == "neglogfu":
        fu = np.power(10.0, -pred)
        return np.log10(np.clip(fu, clip, 1.0 - clip))
    if target == "logitfu":
        fu = 1.0 / (1.0 + np.exp(-pred))
        return np.log10(np.clip(fu, clip, 1.0 - clip))
    raise ValueError(f"Unknown target: {target}")


def ppb_from_target_tensor(pred: torch.Tensor, target: str, clip: float) -> torch.Tensor:
    if target == "activity":
        fu = torch.pow(torch.tensor(10.0, device=pred.device), pred.float())
    elif target == "neglogfu":
        fu = torch.pow(torch.tensor(10.0, device=pred.device), -pred.float())
    elif target == "logitfu":
        fu = torch.sigmoid(pred.float())
    else:
        raise ValueError(f"Unknown target: {target}")
    fu = torch.clamp(fu, clip, 1.0 - clip)
    return torch.clamp((1.0 - fu) / 0.999, 0.0, 1.0)


def sample_weights_from_activity(activity: torch.Tensor, bin_weights: torch.Tensor | None) -> torch.Tensor:
    if bin_weights is None:
        return torch.ones_like(activity, dtype=torch.float32)
    ppb = (1.0 - activity_to_fu(activity)) / 0.999
    ppb = torch.clamp(ppb, 0.0, 1.0)
    edges = torch.tensor(PPB_BINS, dtype=torch.float32, device=activity.device)
    idx = torch.bucketize(ppb.float(), edges[1:-1])
    return bin_weights.to(activity.device)[idx]


def make_bin_weights(train_activity: np.ndarray, mode: str, power: float) -> np.ndarray | None:
    if mode == "none":
        return None
    ppb = activity_to_ppb(train_activity)
    counts = np.histogram(ppb, bins=PPB_BINS)[0].astype(float)
    counts = np.maximum(counts, 1.0)
    if mode == "inverse":
        weights = len(ppb) / (len(counts) * counts)
    elif mode == "sqrt_inverse":
        weights = np.sqrt(len(ppb) / (len(counts) * counts))
    elif mode == "power_inverse":
        weights = (len(ppb) / (len(counts) * counts)) ** power
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    weights = weights / np.mean(weights)
    return weights.astype(np.float32)


def make_sample_weights(train_activity: np.ndarray, mode: str, power: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    bin_weights = make_bin_weights(train_activity, mode, power)
    if bin_weights is None:
        return None, None
    ppb = activity_to_ppb(train_activity)
    bin_idx = np.digitize(ppb, PPB_BINS[1:-1], right=False)
    sample_weights = bin_weights[bin_idx]
    sample_weights = sample_weights / np.mean(sample_weights)
    return sample_weights.astype(np.float64), bin_weights


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, loss_name: str, huber_delta: float):
    mask = ~torch.isnan(target)
    pred = pred[mask]
    target = target[mask]
    weights = weights[mask]
    if loss_name == "mse":
        loss = (pred - target) ** 2
    elif loss_name == "mae":
        loss = torch.abs(pred - target)
    elif loss_name == "huber":
        loss = torch.nn.functional.huber_loss(pred, target, reduction="none", delta=huber_delta)
    elif loss_name == "smoothl1":
        loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=huber_delta)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
    return (loss * weights).sum() / torch.clamp(weights.sum(), min=1e-8)


def low_ppb_pairwise_ranking_loss(
    pred_target: torch.Tensor,
    true_activity: torch.Tensor,
    target: str,
    clip: float,
    threshold: float,
    min_delta: float,
    temperature: float,
) -> torch.Tensor:
    true_ppb = activity_to_ppb(true_activity)
    mask = true_ppb < threshold
    if int(mask.sum().detach().cpu().item()) < 3:
        return pred_target.new_tensor(0.0)

    true_low = true_ppb[mask].float()
    pred_low = ppb_from_target_tensor(pred_target[mask], target, clip).float()
    true_diff = true_low[:, None] - true_low[None, :]
    pred_diff = pred_low[:, None] - pred_low[None, :]
    pair_mask = true_diff.abs() >= min_delta
    pair_mask = torch.triu(pair_mask, diagonal=1)
    if int(pair_mask.sum().detach().cpu().item()) == 0:
        return pred_target.new_tensor(0.0)

    sign = torch.sign(true_diff[pair_mask])
    ordered_pred_diff = pred_diff[pair_mask] * sign
    pair_weight = true_diff[pair_mask].abs()
    pair_loss = torch.nn.functional.softplus(-ordered_pred_diff / max(temperature, 1e-6))
    return (pair_loss * pair_weight).sum() / torch.clamp(pair_weight.sum(), min=1e-8)


def make_aux_bin_head(hidden_channels: int, aux_type: str, dropout: float) -> torch.nn.Module | None:
    if aux_type == "none":
        return None
    out_dim = 1 if aux_type == "binary_low" else len(PPB_BIN_LABELS)
    return torch.nn.Sequential(
        torch.nn.LayerNorm(hidden_channels),
        torch.nn.Linear(hidden_channels, hidden_channels // 2),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_channels // 2, out_dim),
    )


def make_aux_class_weights(train_activity: np.ndarray, aux_type: str, low_threshold: float, power: float):
    if aux_type == "none":
        return None
    ppb = activity_to_ppb(train_activity)
    if aux_type == "binary_low":
        positives = max(float((ppb < low_threshold).sum()), 1.0)
        negatives = max(float((ppb >= low_threshold).sum()), 1.0)
        return np.array([negatives / positives], dtype=np.float32)
    if aux_type == "multiclass":
        counts = np.histogram(ppb, bins=PPB_BINS)[0].astype(float)
        counts = np.maximum(counts, 1.0)
        weights = (len(ppb) / (len(counts) * counts)) ** power
        return (weights / np.mean(weights)).astype(np.float32)
    raise ValueError(f"Unknown auxiliary bin loss type: {aux_type}")


def aux_bin_loss(
    aux_head: torch.nn.Module | None,
    mol_feature: torch.Tensor,
    true_activity: torch.Tensor,
    aux_type: str,
    low_threshold: float,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    if aux_head is None or aux_type == "none":
        return mol_feature.new_tensor(0.0)
    true_ppb = activity_to_ppb(true_activity).float()
    logits = aux_head(mol_feature)
    if aux_type == "binary_low":
        target = (true_ppb < low_threshold).float().view(-1, 1)
        pos_weight = class_weights.to(mol_feature.device) if class_weights is not None else None
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    if aux_type == "multiclass":
        edges = torch.tensor(PPB_BINS, dtype=torch.float32, device=true_ppb.device)
        target = torch.bucketize(true_ppb, edges[1:-1]).long()
        weights = class_weights.to(mol_feature.device) if class_weights is not None else None
        return torch.nn.functional.cross_entropy(logits, target, weight=weights)
    raise ValueError(f"Unknown auxiliary bin loss type: {aux_type}")


def forward_model(model, data, device: str):
    data = data.to(device)
    reg_pred, _, _, attention_dict = model(
        mol_x=data.mol_x,
        mol_x_feat=data.mol_x_feat,
        bond_x=data.mol_edge_attr,
        atom_edge_index=data.mol_edge_index,
        clique_x=data.clique_x,
        clique_edge_index=data.clique_edge_index,
        atom2clique_index=data.atom2clique_index,
        mol_batch=data.mol_x_batch,
        clique_batch=data.clique_x_batch,
    )
    return reg_pred.squeeze(), data, attention_dict


def compute_metrics(true_activity: np.ndarray, pred_activity: np.ndarray, prefix: str) -> dict:
    true_fu = activity_to_fu(true_activity)
    pred_fu = activity_to_fu(pred_activity)
    true_ppb = activity_to_ppb(true_activity)
    pred_ppb = activity_to_ppb(pred_activity)
    ppb_err = pred_ppb - true_ppb
    fu_err = pred_fu - true_fu
    row = {
        f"{prefix}_activity_rmse": float(mean_squared_error(true_activity, pred_activity, squared=False)),
        f"{prefix}_activity_mae": float(mean_absolute_error(true_activity, pred_activity)),
        f"{prefix}_activity_r2": float(r2_score(true_activity, pred_activity)),
        f"{prefix}_ppb_bias": float(ppb_err.mean()),
        f"{prefix}_ppb_mae": float(np.abs(ppb_err).mean()),
        f"{prefix}_ppb_rmse": float(np.sqrt(np.mean(ppb_err**2))),
        f"{prefix}_fu_bias": float(fu_err.mean()),
        f"{prefix}_fu_mae": float(np.abs(fu_err).mean()),
        f"{prefix}_fu_rmse": float(np.sqrt(np.mean(fu_err**2))),
        f"{prefix}_true_ppb_mean": float(true_ppb.mean()),
        f"{prefix}_pred_ppb_mean": float(pred_ppb.mean()),
        f"{prefix}_true_fu_mean": float(true_fu.mean()),
        f"{prefix}_pred_fu_mean": float(pred_fu.mean()),
        f"{prefix}_overpredict_ppb_frac": float((ppb_err > 0).mean()),
    }
    for lo, hi, label in zip(PPB_BINS[:-1], PPB_BINS[1:], PPB_BIN_LABELS):
        mask = (true_ppb >= lo) & (true_ppb < hi)
        if mask.any():
            row[f"{prefix}_bin_{label}_n"] = int(mask.sum())
            row[f"{prefix}_bin_{label}_ppb_bias"] = float(ppb_err[mask].mean())
            row[f"{prefix}_bin_{label}_ppb_mae"] = float(np.abs(ppb_err[mask]).mean())
            row[f"{prefix}_bin_{label}_fu_bias"] = float(fu_err[mask].mean())
    return row


def add_selection_metrics(row: dict, bias_lambda: float) -> None:
    for prefix in ("train", "valid", "test"):
        fu_mae = row.get(f"{prefix}_fu_mae")
        ppb_mae = row.get(f"{prefix}_ppb_mae")
        ppb_bias = row.get(f"{prefix}_ppb_bias")
        if fu_mae is not None and ppb_bias is not None:
            row[f"{prefix}_fu_mae_abs_ppb_bias"] = float(fu_mae + bias_lambda * abs(ppb_bias))
        if ppb_mae is not None and ppb_bias is not None:
            row[f"{prefix}_ppb_mae_abs_ppb_bias"] = float(ppb_mae + bias_lambda * abs(ppb_bias))


def predict(model, loader, device: str, target: str, clip: float) -> pd.DataFrame:
    model.eval()
    smiles = []
    true_activity = []
    pred_target = []
    with torch.no_grad():
        for data in loader:
            pred, data, _ = forward_model(model, data, device)
            smiles.extend(list(data.mol_key))
            true_activity.append(data.reg_y.detach().cpu().numpy().reshape(-1))
            pred_target.append(pred.detach().cpu().numpy().reshape(-1))
    true_activity_np = np.concatenate(true_activity)
    pred_target_np = np.concatenate(pred_target)
    pred_activity_np = activity_from_target(pred_target_np, target, clip)
    out = pd.DataFrame(
        {
            "smiles": smiles,
            "activity": true_activity_np,
            "predicted_target": pred_target_np,
            "predicted_activity": pred_activity_np,
        }
    )
    out["true_fu"] = activity_to_fu(out["activity"].to_numpy())
    out["pred_fu"] = activity_to_fu(out["predicted_activity"].to_numpy())
    out["true_ppb"] = activity_to_ppb(out["activity"].to_numpy())
    out["pred_ppb"] = activity_to_ppb(out["predicted_activity"].to_numpy())
    out["ppb_error"] = out["pred_ppb"] - out["true_ppb"]
    out["fu_error"] = out["pred_fu"] - out["true_fu"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--datafolder", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--config-path", default="config.json")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target", choices=["activity", "neglogfu", "logitfu"], default="neglogfu")
    parser.add_argument("--loss", choices=["mse", "mae", "huber", "smoothl1"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.25)
    parser.add_argument("--low-ppb-rank-lambda", type=float, default=0.0)
    parser.add_argument("--low-ppb-rank-threshold", type=float, default=0.4)
    parser.add_argument("--low-ppb-rank-min-delta", type=float, default=0.03)
    parser.add_argument("--low-ppb-rank-temperature", type=float, default=0.05)
    parser.add_argument("--aux-bin-loss-lambda", type=float, default=0.0)
    parser.add_argument("--aux-bin-loss-type", choices=["none", "binary_low", "multiclass"], default="none")
    parser.add_argument("--aux-bin-low-threshold", type=float, default=0.4)
    parser.add_argument("--aux-bin-class-weight-power", type=float, default=0.5)
    parser.add_argument("--aux-bin-dropout", type=float, default=0.1)
    parser.add_argument("--bin-weighting", choices=["none", "inverse", "sqrt_inverse", "power_inverse"], default="sqrt_inverse")
    parser.add_argument("--bin-weight-power", type=float, default=0.25)
    parser.add_argument("--sampler-weighting", choices=["none", "inverse", "sqrt_inverse", "power_inverse"], default="none")
    parser.add_argument("--sampler-weight-power", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--early-stopping", type=int, default=25)
    parser.add_argument("--save-top-k", type=int, default=1)
    parser.add_argument(
        "--select-metric",
        choices=[
            "valid_fu_mae",
            "valid_ppb_mae",
            "valid_activity_rmse",
            "valid_fu_mae_abs_ppb_bias",
            "valid_ppb_mae_abs_ppb_bias",
        ],
        default="valid_fu_mae",
    )
    parser.add_argument("--select-bias-lambda", type=float, default=0.5)
    parser.add_argument("--set-layer", default="SetRep")
    parser.add_argument("--skip-train-eval", action="store_true", help="During epoch evaluation, skip full train-set prediction.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    os.chdir(project_root)
    result_path = Path(args.result_path)
    result_path.mkdir(parents=True, exist_ok=True)
    (result_path / "save_model").mkdir(exist_ok=True)

    same_seeds(args.seed)
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["params"]["set_layer"] = args.set_layer

    train_df = pd.read_csv(Path(args.datafolder) / "train.csv")
    valid_df = pd.read_csv(Path(args.datafolder) / "valid.csv")
    test_df = pd.read_csv(Path(args.datafolder) / "test.csv")

    ligand_path = Path(args.datafolder) / "ligand.pt"
    if ligand_path.exists():
        ligand_dict = torch.load(str(ligand_path), map_location="cpu")
    else:
        ligand_smiles = list(set(train_df["smiles"].tolist() + valid_df["smiles"].tolist() + test_df["smiles"].tolist()))
        ligand_dict = ligand_init(ligand_smiles)
        torch.save(ligand_dict, str(ligand_path))

    train_dataset = MotifMoleculeDataset(train_df, ligand_dict, device=args.device)
    valid_dataset = MotifMoleculeDataset(valid_df, ligand_dict, device=args.device)
    test_dataset = MotifMoleculeDataset(test_df, ligand_dict, device=args.device)
    sampler_weights_np, sampler_bin_weights_np = make_sample_weights(
        train_df["activity"].to_numpy(float), args.sampler_weighting, args.sampler_weight_power
    )
    train_sampler = None
    if sampler_weights_np is not None:
        train_sampler = CustomWeightedRandomSampler(
            weights=torch.as_tensor(sampler_weights_np, dtype=torch.double),
            num_samples=len(sampler_weights_np),
            replacement=True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        follow_batch=["mol_x", "clique_x"],
    )
    degree_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, follow_batch=["mol_x", "clique_x"])
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, follow_batch=["mol_x", "clique_x"])
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, follow_batch=["mol_x", "clique_x"])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, follow_batch=["mol_x", "clique_x"])

    degree_path = result_path / "degree.pt"
    if degree_path.exists():
        degree_dict = torch.load(str(degree_path), map_location="cpu")
        mol_deg, clique_deg = degree_dict["ligand_deg"], degree_dict["clique_deg"]
    else:
        mol_deg, clique_deg = compute_pna_degrees(degree_loader)
        torch.save({"ligand_deg": mol_deg, "clique_deg": clique_deg}, str(degree_path))

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
    model.reset_parameters()
    aux_head = make_aux_bin_head(config["params"]["hidden_channels"], args.aux_bin_loss_type, args.aux_bin_dropout)
    if aux_head is not None:
        aux_head = aux_head.to(args.device)
    optimizer_params = list(model.parameters())
    if aux_head is not None:
        optimizer_params += list(aux_head.parameters())
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, betas=(0.9, 0.999), eps=args.eps, weight_decay=args.weight_decay)

    bin_weights_np = make_bin_weights(train_df["activity"].to_numpy(float), args.bin_weighting, args.bin_weight_power)
    bin_weights = None if bin_weights_np is None else torch.tensor(bin_weights_np, dtype=torch.float32)
    aux_class_weights_np = make_aux_class_weights(
        train_df["activity"].to_numpy(float),
        args.aux_bin_loss_type,
        args.aux_bin_low_threshold,
        args.aux_bin_class_weight_power,
    )
    aux_class_weights = None if aux_class_weights_np is None else torch.tensor(aux_class_weights_np, dtype=torch.float32)

    metadata = vars(args).copy()
    metadata["bin_weights"] = None if bin_weights_np is None else dict(zip(PPB_BIN_LABELS, [float(x) for x in bin_weights_np]))
    metadata["sampler_bin_weights"] = (
        None if sampler_bin_weights_np is None else dict(zip(PPB_BIN_LABELS, [float(x) for x in sampler_bin_weights_np]))
    )
    metadata["aux_class_weights"] = None if aux_class_weights_np is None else [float(x) for x in aux_class_weights_np]
    (result_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history = []
    topk_records = []
    best_metric = math.inf
    best_epoch = -1
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if aux_head is not None:
            aux_head.train()
        losses = []
        for data in train_loader:
            optimizer.zero_grad()
            pred, data, attention_dict = forward_model(model, data, args.device)
            true_target = target_from_activity(data.reg_y.squeeze(), args.target, args.clip)
            weights = sample_weights_from_activity(data.reg_y.squeeze(), bin_weights)
            loss = weighted_loss(pred.squeeze(), true_target.squeeze(), weights.squeeze(), args.loss, args.huber_delta)
            if args.low_ppb_rank_lambda > 0:
                rank_loss = low_ppb_pairwise_ranking_loss(
                    pred.squeeze(),
                    data.reg_y.squeeze(),
                    args.target,
                    args.clip,
                    args.low_ppb_rank_threshold,
                    args.low_ppb_rank_min_delta,
                    args.low_ppb_rank_temperature,
                )
                loss = loss + args.low_ppb_rank_lambda * rank_loss
            if args.aux_bin_loss_lambda > 0 and aux_head is not None:
                aux_loss = aux_bin_loss(
                    aux_head,
                    attention_dict["mol_feature"],
                    data.reg_y.squeeze(),
                    args.aux_bin_loss_type,
                    args.aux_bin_low_threshold,
                    aux_class_weights,
                )
                loss = loss + args.aux_bin_loss_lambda * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.temperature_clamp()
            losses.append(float(loss.detach().cpu().item()))

        if epoch % args.evaluate_every == 0:
            row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            eval_loaders = [("valid", valid_loader), ("test", test_loader)]
            if not args.skip_train_eval:
                eval_loaders.insert(0, ("train", train_eval_loader))
            for name, loader in eval_loaders:
                pred_df = predict(model, loader, args.device, args.target, args.clip)
                row.update(compute_metrics(pred_df["activity"].to_numpy(float), pred_df["predicted_activity"].to_numpy(float), name))
            add_selection_metrics(row, args.select_bias_lambda)
            history.append(row)
            pd.DataFrame(history).to_csv(result_path / "epoch_metrics.csv", index=False)

            select_value = row[args.select_metric]
            if args.save_top_k > 0:
                epoch_model_path = result_path / "save_model" / f"model_epoch_{epoch:03d}.pt"
                torch.save(model.state_dict(), epoch_model_path)
                if aux_head is not None:
                    torch.save(aux_head.state_dict(), result_path / "save_model" / f"aux_head_epoch_{epoch:03d}.pt")
                topk_records.append({"epoch": epoch, "metric": float(select_value), "path": str(epoch_model_path)})
                topk_records = sorted(topk_records, key=lambda x: x["metric"])
                for stale in topk_records[args.save_top_k :]:
                    stale_path = Path(stale["path"])
                    if stale_path.exists():
                        stale_path.unlink()
                    stale_aux_path = stale_path.with_name(stale_path.name.replace("model_epoch_", "aux_head_epoch_"))
                    if stale_aux_path.exists():
                        stale_aux_path.unlink()
                topk_records = topk_records[: args.save_top_k]
                with open(result_path / "topk_checkpoints.json", "w", encoding="utf-8") as f:
                    json.dump(topk_records, f, indent=2)

            improved = select_value < best_metric
            if improved:
                best_metric = select_value
                best_epoch = epoch
                patience = 0
                torch.save(model.state_dict(), result_path / "save_model" / "model.pt")
                if aux_head is not None:
                    torch.save(aux_head.state_dict(), result_path / "save_model" / "aux_head.pt")
            else:
                patience += 1

            print(
                f"epoch={epoch:03d} loss={row['train_loss']:.5f} "
                f"valid_fu_mae={row['valid_fu_mae']:.5f} valid_ppb_mae={row['valid_ppb_mae']:.5f} "
                f"valid_ppb_bias={row['valid_ppb_bias']:.5f} test_fu_mae={row['test_fu_mae']:.5f} "
                f"test_ppb_bias={row['test_ppb_bias']:.5f} best_epoch={best_epoch}",
                flush=True,
            )
            if patience >= args.early_stopping:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    model.load_state_dict(torch.load(result_path / "save_model" / "model.pt", map_location=args.device))
    best_rows = []
    valid_pred = predict(model, valid_loader, args.device, args.target, args.clip)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(valid_pred["pred_ppb"].to_numpy(float), valid_pred["true_ppb"].to_numpy(float))

    for name, loader in [("train", train_eval_loader), ("valid", valid_loader), ("test", test_loader)]:
        pred_df = predict(model, loader, args.device, args.target, args.clip)
        pred_df["calibrated_ppb"] = iso.predict(pred_df["pred_ppb"].to_numpy(float))
        pred_df["calibrated_fu"] = np.clip(1.0 - 0.999 * pred_df["calibrated_ppb"].to_numpy(float), args.clip, 1.0)
        pred_df["calibrated_activity"] = np.log10(pred_df["calibrated_fu"].to_numpy(float))
        pred_df.to_csv(result_path / f"{name}_predictions_best.csv", index=False)
        raw_metrics = compute_metrics(pred_df["activity"].to_numpy(float), pred_df["predicted_activity"].to_numpy(float), f"{name}_raw")
        cal_metrics = compute_metrics(pred_df["activity"].to_numpy(float), pred_df["calibrated_activity"].to_numpy(float), f"{name}_cal")
        best_rows.append({"split": name, **raw_metrics, **cal_metrics})

    summary = pd.DataFrame(best_rows)
    summary.to_csv(result_path / "best_metrics_with_calibration.csv", index=False)
    with open(result_path / "best_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "best_metric": best_metric, "select_metric": args.select_metric}, f, indent=2)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
