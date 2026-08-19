#!/usr/bin/env python3
"""Export aggregate target-feature references without redistributing training rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-x", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.training_x)
    target_ids = list(frame.columns[3:197])
    if len(target_ids) != 194:
        raise ValueError(f"Expected 194 target columns, found {len(target_ids)}")
    output = pd.DataFrame({
        "target_uniprot": target_ids,
        "training_median": frame[target_ids].median(axis=0).to_numpy(float),
        "training_q90": frame[target_ids].quantile(0.90, axis=0).to_numpy(float),
        "reference_source": "base_random_train_X",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {output.shape}")


if __name__ == "__main__":
    main()
