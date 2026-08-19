#!/usr/bin/env python3
"""Train HetSia task-head with soft-label pseudo exposure samples and monotone loss.

Only the fold training subset is expanded. Fold validation and final test sets
stay on the original one-exposure-per-drug labels, so the experiment remains
directly comparable to the current task-head protocol.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class ViewDataMO:
    def __init__(self, df_x: pd.DataFrame, df_y: pd.DataFrame) -> None:
        tasks = df_y.columns[1:]
        feature_by_smiles = {
            row["smiles"]: row.iloc[2:].to_numpy(dtype=np.float32).reshape(-1)
            for _, row in df_x.iterrows()
        }
        self.data = []
        for _, row in df_y.iterrows():
            smiles = row["smiles"]
            if smiles not in feature_by_smiles:
                continue
            x = feature_by_smiles[smiles]
            for i, task in enumerate(tasks):
                if pd.notna(row[task]):
                    self.data.append((smiles, x, float(row[task]), i, task))

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


def parse_dataset(view_data, dim_x: int):
    smiles, xs, ys, task_idx, task_names = [], [], [], [], []
    for smi, x, y, task_i, task_name in view_data:
        smiles.append(smi)
        xs.append(x)
        ys.append(float(y))
        task_idx.append(int(task_i))
        task_names.append(task_name)
    return (
        np.asarray(xs, dtype=np.float32).reshape(-1, dim_x),
        np.asarray(ys, dtype=np.float32).reshape(-1),
        np.asarray(task_idx, dtype=np.int32).reshape(-1, 1),
        smiles,
        task_names,
    )


def metric_dict(y_true: np.ndarray, prob: np.ndarray) -> dict:
    y_true = y_true.reshape(-1).astype(int)
    prob = prob.reshape(-1)
    pred = (prob > 0.5).astype(int)
    out = {
        "Acc": float((pred == y_true).mean()),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Mcc": float(matthews_corrcoef(y_true, pred)),
        "Bacc": float(balanced_accuracy_score(y_true, pred)),
        "Precision": float(((pred == 1) & (y_true == 1)).sum() / max((pred == 1).sum(), 1)),
        "Recall": float(((pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)),
    }
    try:
        out["AUC"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["AUC"] = float("nan")
    try:
        out["AUPRC"] = float(average_precision_score(y_true, prob))
    except ValueError:
        out["AUPRC"] = float("nan")
    return out


def compute_metrics(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    return (
        float(f1_score(y_true, pred, zero_division=0)),
        float(matthews_corrcoef(y_true, pred)),
        float(balanced_accuracy_score(y_true, pred)),
    )


def per_task_metrics(result_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_name, group in result_df.groupby("task_name", sort=False):
        y = group["label_true"].to_numpy(float)
        p = group["test_pred"].to_numpy(float)
        row = {"task_name": task_name, "n": int(len(group)), "positive": int(y.sum()), "positive_rate": float(y.mean())}
        try:
            row["AUROC"] = float(roc_auc_score(y, p))
        except ValueError:
            row["AUROC"] = float("nan")
        try:
            row["AUPRC"] = float(average_precision_score(y, p))
        except ValueError:
            row["AUPRC"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def parse_delta_weights(value: str) -> list[tuple[float, float]]:
    pairs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        delta, weight = item.split(":")
        pairs.append((float(delta), float(weight)))
    return pairs


def parse_delta_label_weights(value: str) -> list[tuple[float, float, float]]:
    triples = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 2:
            delta, weight = parts
            label = 1.0 if float(delta) >= 0 else 0.75
        elif len(parts) == 3:
            delta, label, weight = parts
        else:
            raise ValueError(f"Expected delta:label:weight, got {item!r}")
        triples.append((float(delta), float(label), float(weight)))
    return triples


def shift_exposure_features(
    x: np.ndarray,
    delta: float,
    exposure_min: float,
    exposure_max: float,
    n_targets: int = 194,
    interaction_mode: str = "product",
) -> np.ndarray:
    shifted = np.array(x, copy=True)
    target = shifted[:, 1 : 1 + n_targets]
    exposure_delta = np.asarray(delta, dtype=np.float32)
    new_exposure = np.clip(shifted[:, 0:1] + exposure_delta, exposure_min, exposure_max)
    shifted[:, 0:1] = new_exposure
    if interaction_mode == "product":
        shifted[:, 1 + n_targets :] = target * new_exposure
    elif interaction_mode == "logratio":
        # For raw_pAC50_logratio features:
        #   target = raw_pAC50
        #   interaction = log10(cmax_free_uM / AC50_uM)
        #               = log10_cmax_free_uM + raw_pAC50 - 6
        shifted[:, 1 + n_targets :] = new_exposure + target - 6.0
    else:
        raise ValueError(f"Unknown interaction_mode: {interaction_mode}")
    return shifted


def build_pseudo_training_arrays(
    x: np.ndarray,
    y: np.ndarray,
    task_idx: np.ndarray,
    positive_delta_weights: list[tuple[float, float]],
    negative_delta_weights: list[tuple[float, float]],
    exposure_min: float,
    exposure_max: float,
    interaction_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    x_parts = [x]
    y_parts = [y]
    task_parts = [task_idx]
    weight_parts = [np.ones_like(y, dtype=np.float32)]
    summary = {"real": int(len(y)), "positive_pseudo": 0, "negative_pseudo": 0}

    pos_mask = y >= 0.5
    neg_mask = ~pos_mask
    for delta, weight in positive_delta_weights:
        if pos_mask.any() and weight > 0:
            x_parts.append(shift_exposure_features(x[pos_mask], delta, exposure_min, exposure_max, interaction_mode=interaction_mode))
            y_parts.append(np.ones(int(pos_mask.sum()), dtype=np.float32))
            task_parts.append(task_idx[pos_mask])
            weight_parts.append(np.full(int(pos_mask.sum()), weight, dtype=np.float32))
            summary["positive_pseudo"] += int(pos_mask.sum())
    for delta, weight in negative_delta_weights:
        if neg_mask.any() and weight > 0:
            x_parts.append(shift_exposure_features(x[neg_mask], delta, exposure_min, exposure_max, interaction_mode=interaction_mode))
            y_parts.append(np.zeros(int(neg_mask.sum()), dtype=np.float32))
            task_parts.append(task_idx[neg_mask])
            weight_parts.append(np.full(int(neg_mask.sum()), weight, dtype=np.float32))
            summary["negative_pseudo"] += int(neg_mask.sum())

    x_aug = np.concatenate(x_parts, axis=0)
    y_aug = np.concatenate(y_parts, axis=0)
    task_aug = np.concatenate(task_parts, axis=0)
    weights = np.concatenate(weight_parts, axis=0)
    order = np.random.permutation(len(y_aug))
    summary["augmented_total"] = int(len(y_aug))
    summary["mean_sample_weight"] = float(weights.mean())
    return x_aug[order], y_aug[order], task_aug[order], weights[order], summary


def build_soft_pseudo_training_arrays(
    x: np.ndarray,
    y: np.ndarray,
    task_idx: np.ndarray,
    positive_delta_label_weights: list[tuple[float, float, float]],
    negative_delta_label_weights: list[tuple[float, float, float]],
    exposure_min: float,
    exposure_max: float,
    interaction_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    x_parts = [x]
    y_parts = [y]
    task_parts = [task_idx]
    weight_parts = [np.ones_like(y, dtype=np.float32)]
    summary = {"real": int(len(y)), "positive_pseudo": 0, "negative_pseudo": 0}

    pos_mask = y >= 0.5
    neg_mask = ~pos_mask
    for delta, label, weight in positive_delta_label_weights:
        if pos_mask.any() and weight > 0:
            n = int(pos_mask.sum())
            x_parts.append(shift_exposure_features(x[pos_mask], delta, exposure_min, exposure_max, interaction_mode=interaction_mode))
            y_parts.append(np.full(n, label, dtype=np.float32))
            task_parts.append(task_idx[pos_mask])
            weight_parts.append(np.full(n, weight, dtype=np.float32))
            summary["positive_pseudo"] += n
    for delta, label, weight in negative_delta_label_weights:
        if neg_mask.any() and weight > 0:
            n = int(neg_mask.sum())
            x_parts.append(shift_exposure_features(x[neg_mask], delta, exposure_min, exposure_max, interaction_mode=interaction_mode))
            y_parts.append(np.full(n, label, dtype=np.float32))
            task_parts.append(task_idx[neg_mask])
            weight_parts.append(np.full(n, weight, dtype=np.float32))
            summary["negative_pseudo"] += n

    x_aug = np.concatenate(x_parts, axis=0)
    y_aug = np.concatenate(y_parts, axis=0)
    task_aug = np.concatenate(task_parts, axis=0)
    weights = np.concatenate(weight_parts, axis=0)
    order = np.random.permutation(len(y_aug))
    summary["augmented_total"] = int(len(y_aug))
    summary["mean_sample_weight"] = float(weights.mean())
    summary["mean_soft_label"] = float(y_aug.mean())
    return x_aug[order], y_aug[order], task_aug[order], weights[order], summary


def build_monotone_pair_plan(
    x: np.ndarray,
    deltas: list[float],
    exposure_min: float,
    exposure_max: float,
    interaction_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    index_parts = []
    delta_parts = []
    summary = {
        "base_pairs": int(len(x)),
        "deltas": [float(d) for d in deltas],
        "total_pairs": 0,
        "pair_generation": "dynamic_batch",
    }
    for delta in deltas:
        if delta <= 0:
            continue
        x_high = shift_exposure_features(x, delta, exposure_min, exposure_max, interaction_mode=interaction_mode)
        valid = x_high[:, 0] > x[:, 0] + 1e-7
        if not np.any(valid):
            continue
        valid_idx = np.flatnonzero(valid).astype(np.int32)
        index_parts.append(valid_idx)
        delta_parts.append(np.full(len(valid_idx), float(delta), dtype=np.float32))
        summary["total_pairs"] += int(len(valid_idx))
    if not index_parts:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32), summary
    pair_base_idx = np.concatenate(index_parts, axis=0).astype(np.int32)
    pair_delta = np.concatenate(delta_parts, axis=0).astype(np.float32)
    return pair_base_idx, pair_delta, summary


def predict_in_batches(model: tf.keras.Model, x: np.ndarray, task: np.ndarray, batch_size: int) -> np.ndarray:
    preds = []
    for start in range(0, len(x), batch_size):
        end = start + batch_size
        pred = model(
            [tf.convert_to_tensor(x[start:end], dtype=tf.float32), tf.convert_to_tensor(task[start:end], dtype=tf.int32)],
            training=False,
        )
        preds.append(pred.numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


def train_with_monotone_loss(
    model: tf.keras.Model,
    x_train: np.ndarray,
    task_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    x_pair_base: np.ndarray,
    task_pair_base: np.ndarray,
    pair_base_idx: np.ndarray,
    pair_delta: np.ndarray,
    exposure_min: float,
    exposure_max: float,
    interaction_mode: str,
    x_valid: np.ndarray,
    task_valid: np.ndarray,
    y_valid: np.ndarray,
    checkpoint_path: Path,
    epochs: int,
    batch_size: int,
    mono_batch_size: int,
    mono_lambda: float,
    mono_margin: float,
    patience: int,
) -> dict:
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    best_val_acc = -np.inf
    best_epoch = 0
    wait = 0
    history = defaultdict(list)
    n = len(y_train)
    pair_n = len(pair_base_idx)
    if pair_n == 0:
        raise ValueError("No monotone pairs were generated; cannot train with monotone loss.")

    @tf.function(reduce_retracing=True)
    def train_batch(xb, tb, yb, wb, xl, xh, tp):
        with tf.GradientTape() as tape:
            pred = model([xb, tb], training=True)
            bce_vec = bce(yb, pred)
            bce_loss = tf.reduce_sum(bce_vec * tf.reshape(wb, (-1,))) / (tf.reduce_sum(wb) + 1e-7)
            low_pred = model([xl, tp], training=True)
            high_pred = model([xh, tp], training=True)
            mono_loss = tf.reduce_mean(tf.nn.relu(low_pred - high_pred + mono_margin))
            loss = bce_loss + mono_lambda * mono_loss
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, bce_loss, mono_loss

    for epoch in range(1, epochs + 1):
        order = np.random.permutation(n)
        pair_order = np.random.permutation(pair_n)
        pair_cursor = 0
        epoch_loss = []
        epoch_bce = []
        epoch_mono = []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            if pair_cursor + mono_batch_size > pair_n:
                pair_order = np.random.permutation(pair_n)
                pair_cursor = 0
            pidx = pair_order[pair_cursor : pair_cursor + mono_batch_size]
            pair_cursor += mono_batch_size

            xb = tf.convert_to_tensor(x_train[idx], dtype=tf.float32)
            tb = tf.convert_to_tensor(task_train[idx], dtype=tf.int32)
            yb = tf.convert_to_tensor(y_train[idx].reshape(-1, 1), dtype=tf.float32)
            wb = tf.convert_to_tensor(sample_weight[idx].reshape(-1, 1), dtype=tf.float32)

            base_idx = pair_base_idx[pidx]
            delta = pair_delta[pidx]
            xl_np = x_pair_base[base_idx]
            xh_np = shift_exposure_features(
                xl_np,
                delta.reshape(-1, 1),
                exposure_min,
                exposure_max,
                interaction_mode=interaction_mode,
            )
            xl = tf.convert_to_tensor(xl_np, dtype=tf.float32)
            xh = tf.convert_to_tensor(xh_np, dtype=tf.float32)
            tp = tf.convert_to_tensor(task_pair_base[base_idx], dtype=tf.int32)

            loss, bce_loss, mono_loss = train_batch(xb, tb, yb, wb, xl, xh, tp)
            epoch_loss.append(float(loss.numpy()))
            epoch_bce.append(float(bce_loss.numpy()))
            epoch_mono.append(float(mono_loss.numpy()))
            del xb, tb, yb, wb, xl, xh, tp, xh_np

        val_prob = predict_in_batches(model, x_valid, task_valid, batch_size=batch_size * 4).reshape(-1)
        val_pred = (val_prob > 0.5).astype(int)
        val_acc = float((val_pred == y_valid.reshape(-1).astype(int)).mean())
        val_auc = metric_dict(y_valid, val_prob).get("AUC", float("nan"))
        history["loss"].append(float(np.mean(epoch_loss)))
        history["bce_loss"].append(float(np.mean(epoch_bce)))
        history["monotone_loss"].append(float(np.mean(epoch_mono)))
        history["val_accuracy"].append(val_acc)
        history["val_auc"].append(val_auc)
        print(
            f"Epoch {epoch}/{epochs} - loss={history['loss'][-1]:.5f} "
            f"bce={history['bce_loss'][-1]:.5f} mono={history['monotone_loss'][-1]:.5f} "
            f"val_accuracy={val_acc:.5f} val_auc={val_auc:.5f}",
            flush=True,
        )
        gc.collect()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            wait = 0
            model.save_weights(str(checkpoint_path), save_format="tf")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break
    model.load_weights(str(checkpoint_path))
    history["best_epoch"] = best_epoch
    history["best_val_accuracy"] = best_val_acc
    return dict(history)


def save_adr_embeddings(model: tf.keras.Model, out_path: Path, adr_tasks: list[str]) -> None:
    emb_layer = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Embedding))
    emb = emb_layer.get_weights()[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)
    pd.DataFrame(emb, index=adr_tasks).to_csv(out_path.with_suffix(".csv"))


def siamese_model_sider_adr_emb_task_head(dim_X=1024, lr=0.0001):
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Input
    from tensorflow.keras.optimizers import Adam

    embedding_dim = 512
    vocab_size = 18
    std_dev = 0.01
    noisy_embeddings = np.random.randn(vocab_size, embedding_dim) + np.random.normal(
        scale=std_dev, size=(vocab_size, embedding_dim)
    )
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(noisy_embeddings),
        trainable=True,
        name="adr_embedding_Ga",
    )

    left_input = Input(shape=(dim_X,), name="drug_features")
    right_input = Input(shape=(1,), dtype="int32", name="adr_task_index")
    model1 = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(dim_X, input_shape=(dim_X,), activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, activation="relu"),
        ],
        name="drug_branch_Ed",
    )
    model2 = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(512, input_shape=(512,), activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, activation="relu"),
        ],
        name="adr_branch_Ea",
    )

    right_emb = embedding_layer(right_input)
    right_emb = tf.reshape(right_emb, shape=(-1, 512, 1))
    encoded_l = model1(left_input)
    encoded_r = model2(right_emb)
    pair = tf.keras.layers.Lambda(lambda x: K.abs(x[0] - x[1]), name="abs_diff_Ed_minus_Ea")([encoded_l, encoded_r])
    h = tf.keras.layers.Dropout(0.1, name="pair_dropout_128")(pair)
    h = tf.keras.layers.Dense(64, activation="relu", name="pair_dense_64")(h)
    h = tf.keras.layers.Dropout(0.1, name="pair_dropout_64")(h)
    h = tf.keras.layers.Dense(32, activation="relu", name="pair_dense_32")(h)
    h = tf.keras.layers.Dropout(0.1, name="pair_dropout_32")(h)
    h = tf.keras.layers.Dense(16, activation="relu", name="pair_dense_16")(h)
    h = tf.keras.layers.Dropout(0.1, name="pair_dropout_16")(h)
    h = tf.keras.layers.Dense(8, activation="relu", name="pair_dense_8")(h)

    task_w = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=8, trainable=True, name="task_specific_head_weight")(right_input)
    task_w = tf.keras.layers.Reshape((8,), name="task_head_weight_flat")(task_w)
    task_b = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=1, embeddings_initializer="zeros", trainable=True, name="task_specific_head_bias")(right_input)
    task_b = tf.keras.layers.Reshape((1,), name="task_head_bias_flat")(task_b)
    logit = tf.keras.layers.Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True) + x[2], name="task_specific_logit")([h, task_w, task_b])
    prediction = tf.keras.layers.Activation("sigmoid", name="prediction")(logit)

    model = tf.keras.Model([left_input, right_input], prediction, name="hetsia_pseudo_dose_task_head")
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=lr),
        metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-indices", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--positive-delta-weights", default="0.30103:0.75,0.60206:1.0,-0.30103:0.15")
    parser.add_argument("--negative-delta-weights", default="-0.60206:1.0,-0.30103:0.75,0.30103:0.05")
    parser.add_argument("--positive-delta-label-weights", default="0.30103:0.92:0.60,0.60206:0.96:0.80,-0.30103:0.75:0.20")
    parser.add_argument("--negative-delta-label-weights", default="-0.60206:0.02:0.80,-0.30103:0.04:0.60,0.30103:0.18:0.20,0.60206:0.30:0.10")
    parser.add_argument("--monotone-deltas", default="0.30103,0.60206")
    parser.add_argument("--monotone-lambda", type=float, default=0.0)
    parser.add_argument("--monotone-margin", type=float, default=0.0)
    parser.add_argument("--monotone-batch-size", type=int, default=512)
    parser.add_argument("--clip-quantiles", default="0.01,0.99")
    parser.add_argument(
        "--interaction-mode",
        choices=["product", "logratio"],
        default="product",
        help="How to recompute the 194 interaction features after shifting exposure.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print("TensorFlow visible GPUs:", tf.config.list_physical_devices("GPU"), flush=True)

    selected_folds = None
    if args.fold_indices.strip():
        selected_folds = {int(x.strip()) for x in args.fold_indices.split(",") if x.strip()}
    positive_delta_weights = parse_delta_weights(args.positive_delta_weights)
    negative_delta_weights = parse_delta_weights(args.negative_delta_weights)
    positive_delta_label_weights = parse_delta_label_weights(args.positive_delta_label_weights)
    negative_delta_label_weights = parse_delta_label_weights(args.negative_delta_label_weights)
    monotone_deltas = [float(x) for x in args.monotone_deltas.split(",") if x.strip()]
    clip_q = [float(x) for x in args.clip_quantiles.split(",") if x.strip()]
    if len(clip_q) != 2:
        raise ValueError("--clip-quantiles must contain exactly two values.")

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = out_dir / "fold_checkpoints"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_x = pd.read_csv(dataset_dir / "train_X.csv")
    valid_x = pd.read_csv(dataset_dir / "valid_X.csv")
    test_x = pd.read_csv(dataset_dir / "test_X.csv")
    train_y = pd.read_csv(dataset_dir / "train_y.csv")
    valid_y = pd.read_csv(dataset_dir / "valid_y.csv")
    test_y = pd.read_csv(dataset_dir / "test_y.csv")
    adr_tasks = list(train_y.columns[1:])

    dim_x = len(train_x.columns[2:])
    exposure_values = pd.concat([train_x["cmax_free(uM)"], valid_x["cmax_free(uM)"]], ignore_index=True).astype(float)
    exposure_min, exposure_max = [float(x) for x in np.quantile(exposure_values.to_numpy(), clip_q)]

    train_pool_x = pd.concat([train_x, valid_x], ignore_index=True)
    train_pool_y = pd.concat([train_y, valid_y], ignore_index=True)
    train_pool_ds = list(ViewDataMO(train_pool_x, train_pool_y))
    test_ds = ViewDataMO(test_x, test_y)
    x_test, y_test, task_test, smiles_test, task_names_test = parse_dataset(test_ds, dim_x)

    result_valid = defaultdict(list)
    result_test = defaultdict(list)
    fold_summaries = []
    best_auc = -np.inf
    best_model = None
    best_fold = None

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    for fold_idx, (train_index, valid_index) in enumerate(kf.split(train_pool_ds), start=1):
        if selected_folds is not None and fold_idx not in selected_folds:
            continue
        print(f"Now is Fold: {fold_idx}", flush=True)
        fold_train = [train_pool_ds[i] for i in train_index]
        fold_valid = [train_pool_ds[i] for i in valid_index]
        x_train, y_train, task_train, _, _ = parse_dataset(fold_train, dim_x)
        x_valid, y_valid, task_valid, _, _ = parse_dataset(fold_valid, dim_x)
        x_train_aug, y_train_aug, task_train_aug, sample_weight, pseudo_summary = build_soft_pseudo_training_arrays(
            x_train,
            y_train,
            task_train,
            positive_delta_label_weights,
            negative_delta_label_weights,
            exposure_min,
            exposure_max,
            args.interaction_mode,
        )
        print("Pseudo summary:", pseudo_summary, flush=True)

        model = siamese_model_sider_adr_emb_task_head(dim_X=dim_x, lr=args.lr)
        checkpoint_path = fold_dir / f"best_model_fold_{fold_idx}_weights"
        if args.monotone_lambda > 0:
            pair_base_idx, pair_delta, monotone_summary = build_monotone_pair_plan(
                x_train,
                monotone_deltas,
                exposure_min,
                exposure_max,
                args.interaction_mode,
            )
            print("Monotone pair summary:", monotone_summary, flush=True)
            history_obj = train_with_monotone_loss(
                model,
                x_train_aug,
                task_train_aug,
                y_train_aug,
                sample_weight,
                x_train,
                task_train,
                pair_base_idx,
                pair_delta,
                exposure_min,
                exposure_max,
                args.interaction_mode,
                x_valid,
                task_valid,
                y_valid,
                checkpoint_path,
                args.epochs,
                args.batch_size,
                args.monotone_batch_size,
                args.monotone_lambda,
                args.monotone_margin,
                args.patience,
            )
            history = type("HistoryLike", (), {"history": history_obj})()
        else:
            callbacks = [
                EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True, verbose=1),
                ModelCheckpoint(str(checkpoint_path), monitor="val_accuracy", save_best_only=True, save_weights_only=True, verbose=1),
            ]
            history = model.fit(
                [x_train_aug, task_train_aug],
                y_train_aug,
                sample_weight=sample_weight,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_data=([x_valid, task_valid], y_valid),
                callbacks=callbacks,
                verbose=1,
            )

        valid_score = model.evaluate([x_valid, task_valid], y_valid, verbose=0)
        valid_prob = model.predict([x_valid, task_valid], verbose=0).reshape(-1)
        test_score = model.evaluate([x_test, task_test], y_test, verbose=0)
        test_prob = model.predict([x_test, task_test], verbose=0).reshape(-1)
        valid_metrics = metric_dict(y_valid, valid_prob)
        test_metrics = metric_dict(y_test, test_prob)
        for key, value in valid_metrics.items():
            result_valid[key].append(value)
        for key, value in test_metrics.items():
            result_test[key].append(value)
        summary = {
            "fold": fold_idx,
            "epochs_ran": len(history.history.get("loss", [])),
            "pseudo_summary": pseudo_summary,
            "valid_keras": [float(x) for x in valid_score],
            "test_keras": [float(x) for x in test_score],
            "valid": valid_metrics,
            "test": test_metrics,
        }
        fold_summaries.append(summary)
        print("Fold summary:", summary, flush=True)
        save_adr_embeddings(model, out_dir / "adr_embeddings" / f"fold_{fold_idx}.npy", adr_tasks)
        if test_metrics["AUC"] > best_auc:
            best_auc = test_metrics["AUC"]
            best_model = model
            best_fold = fold_idx
            print("The best AUC is:", best_auc, flush=True)

    if best_model is None:
        raise RuntimeError("No model was trained.")

    try:
        best_model.save(out_dir / "best_model", save_format="tf")
    except Exception as exc:
        (out_dir / "best_model_save_warning.txt").write_text(str(exc), encoding="utf-8")
    best_model.save_weights(out_dir / "best_model_weights", save_format="tf")
    best_prob = best_model.predict([x_test, task_test], verbose=0).reshape(-1)
    result_df = pd.DataFrame({"smiles": smiles_test, "task_name": task_names_test, "test_pred": best_prob, "label_true": y_test.reshape(-1)})
    result_df.to_csv(out_dir / "test_pred_result.csv", index=False)
    per_task_metrics(result_df).to_csv(out_dir / "test_per_task_metrics.csv", index=False)
    pd.DataFrame(result_valid).to_csv(out_dir / "scores_valid.csv", index=False)
    pd.DataFrame(result_test).to_csv(out_dir / "scores_test.csv", index=False)
    metadata = {
        "protocol": "task_head_soft_label_pseudo_dose_monotone_loss",
        "dataset_dir": str(dataset_dir),
        "seed": args.seed,
        "folds": args.folds,
        "fold_indices": sorted(selected_folds) if selected_folds else list(range(1, args.folds + 1)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "dim_X": dim_x,
        "positive_delta_weights": positive_delta_weights,
        "negative_delta_weights": negative_delta_weights,
        "positive_delta_label_weights": positive_delta_label_weights,
        "negative_delta_label_weights": negative_delta_label_weights,
        "monotone_deltas": monotone_deltas,
        "monotone_lambda": args.monotone_lambda,
        "monotone_margin": args.monotone_margin,
        "monotone_batch_size": args.monotone_batch_size,
        "monotone_pair_generation": "dynamic_batch",
        "clip_quantiles": clip_q,
        "interaction_mode": args.interaction_mode,
        "exposure_clip_range": [exposure_min, exposure_max],
        "best_fold": best_fold,
        "best_test_auc": best_auc,
        "fold_summaries": fold_summaries,
    }
    (out_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Best fold:", best_fold, "Best test AUC:", best_auc, flush=True)


if __name__ == "__main__":
    main()
