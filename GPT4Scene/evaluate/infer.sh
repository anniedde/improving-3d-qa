#!/bin/bash

# Check for argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

# Grab experiment name from first argument
experiment_name=$1

echo "Running evaluation for experiment: $experiment_name"

#name="GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512"
name="GPT4Scene-qwen2vl-pretrained"
export PYTHONPATH=.

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
export CHUNKS=${#GPULIST[@]}

echo "Evaluating task in $CHUNKS GPUs"

BASE_ARGS=(
    "./evaluate/configs/eval_configs.py"
    "evaluate" "True"
    "model" "qwenvl"
    "batch_size" "1"
    "num_workers" "16"
    "model_path" "./model_outputs/${name}"
    #"scene_anno" "./evaluate/annotation/selected_images_mark_3D_val_32.json"
    "scene_anno" "./evaluate/annotation/modified/selected_images_${experiment_name}.json"
    "save_interval" "2"
)

tasks=("scanqa") #"sqa3d" "scan2cap" "scanrefer" "multi3dref")

for task in "${tasks[@]}"; do
    echo "Starting task: $task"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        echo "Launching chunk: $IDX for task: $task"
        ARGS=(
            "${BASE_ARGS[@]}"
            "val_tag" "$task"
            #"output_dir" "eval_outputs/${experiment_name}/${name}/"
            "output_dir" "eval_outputs_manual/outputs_3D_mark/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512"
            "num_chunks" "$CHUNKS"
            "chunk_idx" "$IDX"
            "calculate_score_tag" "scanqa"
        )

        #CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluate/infer.py "${ARGS[@]}" &
    done

    wait
    echo "Finished task: $task"
done

wait

CUDA_VISIBLE_DEVICES=0 python evaluate/calculate_scores.py "${ARGS[@]}"

wait