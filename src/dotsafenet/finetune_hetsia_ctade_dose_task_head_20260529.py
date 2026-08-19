#!/usr/bin/env python3
"""Lightweight CT-ADE clinical-dose fine-tuning for HetSia task-head weights."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hetsia_soft_pseudo_dose_task_head import (  # noqa: E402
    ViewDataMO,
    parse_dataset,
    per_task_metrics,
    save_adr_embeddings,
    siamese_model_sider_adr_emb_task_head,
)


SYNTHETIC_TAG = "__ctade_clinical_dose__"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def metric_dict(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y = y_true.reshape(-1).astype(int)
    p = prob.reshape(-1)
    pred = (p > 0.5).astype(int)
    out = {
        "Acc": float((pred == y).mean()),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Mcc": float(matthews_corrcoef(y, pred)),
        "Bacc": float(balanced_accuracy_score(y, pred)),
        "Precision": float(((pred == 1) & (y == 1)).sum() / max((pred == 1).sum(), 1)),
        "Recall": float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
    }
    try:
        out["AUC"] = float(roc_auc_score(y, p))
    except ValueError:
        out["AUC"] = float("nan")
    try:
        out["AUPRC"] = float(average_precision_score(y, p))
    except ValueError:
        out["AUPRC"] = float("nan")
    return out


def load_arrays(dataset_dir: Path, split: str, dim_x: int | None = None):
    x_df = pd.read_csv(dataset_dir / f"{split}_X.csv")
    y_df = pd.read_csv(dataset_dir / f"{split}_y.csv")
    if dim_x is None:
        dim_x = len(x_df.columns[2:])
    ds = ViewDataMO(x_df, y_df)
    return parse_dataset(ds, dim_x)


def sample_weights(smiles: list[str], clinical_weight: float) -> np.ndarray:
    return np.asarray(
        [clinical_weight if SYNTHETIC_TAG in str(smi) else 1.0 for smi in smiles],
        dtype=np.float32,
    )


def set_trainable_scope(model: tf.keras.Model, scope: str) -> None:
    if scope == "all":
        for layer in model.layers:
            layer.trainable = True
        return
    trainable_names = {
        "task_head": {"task_specific_head_weight", "task_specific_head_bias"},
        "task_head_pair8": {"task_specific_head_weight", "task_specific_head_bias", "pair_dense_8"},
        "pair_stack": {
            "task_specific_head_weight",
            "task_specific_head_bias",
            "pair_dense_8",
            "pair_dense_16",
            "pair_dense_32",
            "pair_dense_64",
        },
    }[scope]
    for layer in model.layers:
        layer.trainable = layer.name in trainable_names


def predict_and_write(
    model: tf.keras.Model,
    x: np.ndarray,
    task: np.ndarray,
    y: np.ndarray,
    smiles: list[str],
    task_names: list[str],
    out_path: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    prob = model.predict([x, task], verbose=0).reshape(-1)
    result = pd.DataFrame(
        {
            "smiles": smiles,
            "task_name": task_names,
            "test_pred": prob,
            "label_true": y.reshape(-1),
        }
    )
    result.to_csv(out_path, index=False)
    per_task_metrics(result).to_csv(out_path.with_name(out_path.stem + "_per_task_metrics.csv"), index=False)
    return result, metric_dict(y, prob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--clinical-test-dataset-dir", type=Path, default=None)
    parser.add_argument("--baseline-weights", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--clinical-weight", type=float, default=0.35)
    parser.add_argument("--trainable-scope", choices=["task_head", "task_head_pair8", "pair_stack", "all"], default="task_head")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print("TensorFlow visible GPUs:", tf.config.list_physical_devices("GPU"), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_x_df = pd.read_csv(args.dataset_dir / "train_X.csv")
    dim_x = len(train_x_df.columns[2:])
    train_x, train_y, train_task, train_smiles, _ = load_arrays(args.dataset_dir, "train", dim_x)
    valid_x, valid_y, valid_task, valid_smiles, valid_task_names = load_arrays(args.dataset_dir, "valid", dim_x)
    test_x, test_y, test_task, test_smiles, test_task_names = load_arrays(args.dataset_dir, "test", dim_x)

    model = siamese_model_sider_adr_emb_task_head(dim_X=dim_x, lr=args.lr)
    model.load_weights(args.baseline_weights)
    baseline_result, baseline_metrics = predict_and_write(
        model, test_x, test_task, test_y, test_smiles, test_task_names, args.out_dir / "baseline_test_pred_result.csv"
    )
    set_trainable_scope(model, args.trainable_scope)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(args.out_dir / "best_finetuned_weights"), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
    ]
    weights = sample_weights(train_smiles, args.clinical_weight)
    history = model.fit(
        [train_x, train_task],
        train_y,
        sample_weight=weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=([valid_x, valid_task], valid_y),
        callbacks=callbacks,
        verbose=1,
    )

    _, finetuned_metrics = predict_and_write(
        model, test_x, test_task, test_y, test_smiles, test_task_names, args.out_dir / "finetuned_test_pred_result.csv"
    )
    clinical_metrics = None
    if args.clinical_test_dataset_dir is not None:
        clin_x, clin_y, clin_task, clin_smiles, clin_task_names = load_arrays(args.clinical_test_dataset_dir, "test", dim_x)
        _, clinical_metrics = predict_and_write(
            model,
            clin_x,
            clin_task,
            clin_y,
            clin_smiles,
            clin_task_names,
            args.out_dir / "finetuned_clinical_test_pred_result.csv",
        )

    save_adr_embeddings(model, args.out_dir / "adr_embeddings" / "finetuned.npy", list(pd.read_csv(args.dataset_dir / "train_y.csv").columns[1:]))
    model.save_weights(args.out_dir / "finetuned_model_weights", save_format="tf")
    pd.DataFrame(history.history).to_csv(args.out_dir / "finetune_history.csv", index=False)
    metadata = {
        "dataset_dir": str(args.dataset_dir),
        "clinical_test_dataset_dir": str(args.clinical_test_dataset_dir) if args.clinical_test_dataset_dir else None,
        "baseline_weights": args.baseline_weights,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "clinical_weight": args.clinical_weight,
        "trainable_scope": args.trainable_scope,
        "n_train_task_samples": int(len(train_y)),
        "n_clinical_train_task_samples": int((np.asarray([SYNTHETIC_TAG in str(s) for s in train_smiles])).sum()),
        "baseline_test_metrics": baseline_metrics,
        "finetuned_test_metrics": finetuned_metrics,
        "finetuned_clinical_test_metrics": clinical_metrics,
    }
    (args.out_dir / "finetune_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
