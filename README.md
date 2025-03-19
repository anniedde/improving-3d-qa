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