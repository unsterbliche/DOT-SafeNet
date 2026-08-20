#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?}" "${CLINICAL_TEST_DATASET:?}" "${BASELINE_WEIGHTS_DIR:?}" "${FINETUNED_WEIGHTS_DIR:?}" "${OUTPUT_ROOT:?}"
python "$PROJECT_ROOT/src/dotsafenet/predict_clinical_test_before_after.py" \
  --clinical-test-dataset-dir "$CLINICAL_TEST_DATASET" \
  --baseline-weights-dir "$BASELINE_WEIGHTS_DIR" \
  --finetuned-result-dir "$FINETUNED_WEIGHTS_DIR" \
  --out-dir "$OUTPUT_ROOT"
