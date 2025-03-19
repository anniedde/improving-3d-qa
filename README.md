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