#!/bin/bash
dataset_name=$1
echo "Training on ... ${dataset_name}"
python train_gui.py \
    --source_path ${DATASET_PATH}/${dataset_name} \
    --model_path outputs/${dataset_name}_${EXPERIMENT_NAME} \
    --eval \
    --load2gpu_on_the_fly


