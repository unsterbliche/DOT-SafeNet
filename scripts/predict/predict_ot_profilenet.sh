#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?}" "${OT_DATA_ROOT:?}" "${OT_CHECKPOINT_ROOT:?}" "${INPUT_CSV:?}" "${OUTPUT_JSON:?}" "${DEVICE:?}"
python "$PROJECT_ROOT/src/ot_profilenet/finetuning/predict_drugs.py" \
  --data-root "$OT_DATA_ROOT" \
  --checkpoint-root "$OT_CHECKPOINT_ROOT" \
  --input-csv "$INPUT_CSV" \
  --smiles-column "${SMILES_COLUMN:-smiles}" \
  --output-json "$OUTPUT_JSON" \
  --device "$DEVICE"
