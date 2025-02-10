#!/bin/bash
dataset_name=$1
echo "Training on ... ${dataset_name}"
python train_gui.py \
    --source_path ${DATASET_PATH}/${dataset_name} \
    --model_path outputs/${dataset_name}_${EXPERIMENT_NAME} \
    --deform_type mlp \
    --eval \
    --gt_alpha_mask_as_scene_mask \
    --local_frame \
    --resolution 2 \
    --load2gpu_on_the_fly


