#!/usr/bin/env bash
set -euo pipefail
: "$PROJECT_ROOT" "$CMAX_DATA" "$CMAX_MODEL_CONFIG" "$OUTPUT_ROOT"
for seed in 1998 2022 2023 2024 2025; do
python "$PROJECT_ROOT/src/dose_exponet/main.py" --seed "$seed" --device "$DEVICE" --config_path "$CMAX_MODEL_CONFIG" --datafolder "$CMAX_DATA" --result_path "$OUTPUT_ROOT/run$seed" --regression_task True --dose_mode True --epochs 200 --evaluate_epoch 1 --lrate 1e-4 --eps 1e-5 --betas '(0.9,0.999)' --batch_size 64 --set_layer SetRep --early_stopping_epochs 500
done
