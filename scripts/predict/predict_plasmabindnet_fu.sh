#!/usr/bin/env bash
set -euo pipefail
: "$PROJECT_ROOT" "$INPUT_CSV" "$OUTPUT_CSV" "$CHECKPOINT_DIRS"
python "$PROJECT_ROOT/src/plasmabindnet_fu/predict_ppb_ensemble_for_smiles.py" --motif-root "$PROJECT_ROOT/src/plasmabindnet_fu" --input-csv "$INPUT_CSV" --smiles-col "$SMILES_COLUMN" --checkpoint-dirs "$CHECKPOINT_DIRS" --out-csv "$OUTPUT_CSV" --target logitfu --set-layer SetRep --clip 1e-4 --batch-size 128 --device "$DEVICE"
