#!/bin/bash

# Get experiment name from command line argument
if [ -z "$1" ]; then
    echo "Error: Experiment name argument is required"
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

EXPERIMENT_NAME="$1"

echo "Starting pipeline processing..."

# 1. Extract video features using HERO CLIP (UniVTG environment)
echo "Extracting video features..."
cd HERO_Video_Feature_Extractor/clip \
conda run -n univtg python gather_video_paths.py --video_path ../../scannet_scenes/videos \
chmod +x HERO_Video_Feature_Extractor/clip/run.sh \
conda run -n univtg bash run.sh \

# 2. Process text and extract features (UniVTG environment)
echo "Processing text and extracting features..."
cd improving-3d-qa/UniVTG \
conda run -n univtg python preprocess_eval.py \
conda run -n univtg python transform_prompts.py
conda run -n univtg python batch_encode_text.py --prompts_loc eval_transformed.jsonl --output_dir features/txt

# 3. Run UniVTG for temporal grounding (UniVTG environment)
echo "Running temporal grounding..."
cd improving-3d-qa/UniVTG
conda run -n univtg python infer.sh \

# 4. Run GPT4Scene for 3D understanding (GPT4Scene environment)
echo "Running 3D scene understanding..."
cd improving-3d-qa/GPT4Scene/evaluate
conda run -n gpt4scene bash infer.sh "$EXPERIMENT_NAME"

echo "Pipeline completed. Results saved to $OUTPUT_DIR"

exit 0