# Improving 3D Question Answering via Enhanced Video Frame Sampling

## Pipeline
```ascii
                        Natural Language Question
                                ↓
Video Input               [Query Transformation]
    ↓                     (Subject Extraction)
[HERO Feature Extractor]         ↓
    ↓                     Natural Language Subject
    ↓                            ↓
    ↓                    [Text Feature Extractor]
    ↓                            ↓
Video Features  ──────→    [UniVTG] ← Text Features
                              ↓
                      Relevant Video Clip
                              ↓
                    [Frame Sampling (8 frames)]
                              ↓
Natural Language Question →  [GPT4Scene]
(3D-QA Prompt)                  ↓
                    3D Scene Understanding
                              ↓
                         QA Output
```

### Components
This project is based on multiple repositories that have been forked and adapted. Significant modifications have been made to the original code, along with additional functionality and custom implementations. Specifically, we build upon the codebases of [HERO](https://github.com/linjieli222/HERO_Video_Feature_Extractor), [UniVTG](https://github.com/showlab/UniVTG), and [GPT4Scene](https://github.com/Qi-Zhangyang/GPT4Scene).

1. **Video Feature Extraction**
   - Uses HERO Feature Extractor to process input videos
   - Outputs video features 

2. **Query Processing**
   - Query Transformation: Extracts the subject from the natural language question
   - Text Feature Extractor: Converts the extracted subject into feature embeddings

3. **Temporal Grounding (UniVTG)**
   - Takes video features and text features as input
   - Identifies the most relevant video clip for the given query
   - Then, 8 frames are sampled uniformly from the identified clip

4. **3D Scene Understanding (GPT4Scene)**
   - Takes sampled frames and original question as input

## Installation
Due to dependency requirements, this project requires two separate conda environments for different components.

### 1. UniVTG Environment (Temporal Grounding)
```bash
cd UniVTG
conda create --name univtg python=3.8
pip install -r requirements.txt
```
Download UniVTG checkpoints from Google Drive:
1. Visit: https://drive.google.com/drive/folders/1l6RyjGuqkzfZryCC6xwTZsvjWaIMVxIO
2. Download all files from the folder
3. Place the files as: results/omni/model_best.ckpt and results/omni/opt.json

### 2. GPT4Scene Environment (3D Scene Understanding)
```bash
cd GPT4Scene
conda create --name gpt4scene python=3.10
conda activate gpt4scene

pip install -e ".[torch,metrics]"
```
Sometimes, the PyTorch downloaded this way may encounter errors. In such cases, you need to manually install Pytorch.
```bash
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install qwen_vl_utils flash-attn
```
You can download all trained GPT4Scene model weights, dataset and annotations by running the following command:
```bash
python download.py
```

## Data Preparation

### Download and Preprocess ScanNet
1. Request access to ScanNet v2 dataset from [the official website](http://www.scan-net.org/)
2. Download ScanNet videos.
3. Preprocess raw scans so that the data directory looks like this:
```ascii
improving-3d-qa/
└── scannet_scenes/
    ├── 2d_color_images/
    │   └── scene0000_00/
    │       └── color/
    │           ├── 0.jpg
    │           ├── 1.jpg
    │           ├── 2.jpg
    │           ├── ...
    │           └── 15.jpg
    └── videos/
        ├── scene0000_02.mp4
        ├── scene0001_00.mp4
        ├── scene0001_01.mp4
        ├── ...
        └── scene0005_00.mp4
```

## Usage

Simply run:
```bash
./run_pipeline.sh
```

This script handles:
- Environment switching between UniVTG and GPT4Scene
- Video feature extraction
- Text processing and feature extraction
- Temporal grounding
- 3D scene understanding
- Result aggregation

For detailed pipeline steps and individual component usage, see `run_pipeline.sh`.