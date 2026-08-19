#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?}" "${MODEL_CONFIG:?}" "${MODEL_DATA:?}" "${INPUT_CSV:?}" "${OUTPUT_ROOT:?}" "${PPB_CHECKPOINT:?}" "${CMAX_CHECKPOINT:?}" "${DEVICE:?}"
python "$PROJECT_ROOT/src/dose_exponet/predict_drugs_dose.py" \
  --config_path "$MODEL_CONFIG" \
  --datafolder "$MODEL_DATA" \
  --input-csv "$INPUT_CSV" \
  --smiles-column "${SMILES_COLUMN:-smiles}" \
  --dose-list "${DOSE_LIST:-[10,100]}" \
  --result_path "$OUTPUT_ROOT" \
  --ppb-checkpoint "$PPB_CHECKPOINT" \
  --cmax-checkpoint "$CMAX_CHECKPOINT" \
  --device "$DEVICE"
