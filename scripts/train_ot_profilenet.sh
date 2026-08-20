#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?}" "${OT_DATA_ROOT:?}" "${OT_OUTPUT_ROOT:?}" "${DEVICE:?}"
python "$PROJECT_ROOT/src/ot_profilenet/pretraining/train.py" \
  --data-root "$OT_DATA_ROOT/pretraining" \
  --output-root "$OT_OUTPUT_ROOT/pretraining" \
  --dataset CPI_data_cls --device "$DEVICE" --seed 2024 \
  --train_batch_size 1 --test_batch_size 1 --lr 1e-4 --num_epochs 100

PRETRAINED_CHECKPOINT="${OT_PRETRAINED_CHECKPOINT:-$OT_OUTPUT_ROOT/pretraining/CPI_data_cls/bach1LR0.0001random2024esm2.pt}"
for family in GPCR IonChannel Enzyme Kinase NHR Transporter Other; do
  python "$PROJECT_ROOT/src/ot_profilenet/finetuning/train_reg_fintune.py" \
    --data-root "$OT_DATA_ROOT/target_families" \
    --output-root "$OT_OUTPUT_ROOT/family_checkpoints" \
    --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
    --dataset "$family" --device "$DEVICE" --seed 2024 \
    --train_batch_size 128 --test_batch_size 128 --lr 1e-4 --num_epochs 200
done
