#!/usr/bin/env bash
set -euo pipefail
: "$PROJECT_ROOT" "$PPB_DATA" "$PPB_MODEL_CONFIG" "$OUTPUT_ROOT"
for seed in 2044 2038 2037 2022 2032; do
python "$PROJECT_ROOT/src/plasmabindnet_fu/train_ppb_retraining.py" --project-root "$PROJECT_ROOT/src/plasmabindnet_fu" --datafolder "$PPB_DATA" --result-path "$OUTPUT_ROOT/seed$seed" --config-path "$PPB_MODEL_CONFIG" --seed "$seed" --device "$DEVICE" --target logitfu --loss huber --huber-delta 1.0 --sampler-weighting power_inverse --sampler-weight-power 0.25 --bin-weighting none --aux-bin-loss-type binary_low --aux-bin-loss-lambda 0.05 --aux-bin-low-threshold 0.4 --clip 1e-4 --epochs 60 --batch-size 64 --lr 1e-4 --eps 1e-5 --weight-decay 1e-7 --early-stopping 20 --save-top-k 1 --select-metric valid_fu_mae_abs_ppb_bias --select-bias-lambda 0.5 --set-layer SetRep --skip-train-eval
done
