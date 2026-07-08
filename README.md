# SmallVLM-JEPA Traffic Video Analysis

A modular video understanding pipeline for analyzing traffic scenes,
tracking road users, studying motion, and identifying potentially risky
interactions or accident events.

The project combines visual scene understanding, object tracking, motion
analysis, and interaction reasoning to produce structured traffic video
analysis.

## Overview

The pipeline processes traffic or dashcam videos in multiple stages:

1.  **Scene Understanding** -- extracts a natural-language description
    of the visible scene using SmolVLM2.
2.  **Object Detection and Tracking** -- detects and tracks vehicles and
    pedestrians using YOLO11 and ByteTrack.
3.  **Object Motion Analysis** -- analyzes track quality, apparent
    motion, depth changes, and lateral movement.
4.  **Interaction Analysis** -- evaluates interactions between tracked
    road users and ranks potential traffic conflicts or accident events.
5.  **Structured Reasoning** -- uses Qwen2.5 to combine the extracted
    evidence into a structured JSON analysis.

## Pipeline

``` text
Input Video
    |
    v
Scene Captioning
SmolVLM2
    |
    v
Object Detection + Tracking
YOLO11 + ByteTrack
    |
    v
Object Motion Analysis
    |
    v
Interaction Analysis
    |
    v
Qwen2.5 Fusion
    |
    v
Structured JSON Output
```

## Models and Tools

-   **SmolVLM2** -- visual scene understanding
-   **YOLO11** -- traffic object detection
-   **ByteTrack** -- persistent multi-object tracking
-   **Qwen2.5-3B-Instruct** -- evidence fusion and structured reasoning
-   **OpenCV** -- video and frame processing
-   **PyTorch / Transformers** -- model inference

Models are loaded from the Hugging Face Hub or the Ultralytics model
ecosystem.

## Project Structure

``` text
SmallVLM-JEPA-Captioning/
├── scene_caption.py
├── object_tracker.py
├── object_analyzer.py
├── interaction_analyzer.py
├── action_probe.py
├── fuse_qwen.py
├── pipeline.py
├── requirements.txt
├── videos/
└── results/
```

## Main Components

### `scene_caption.py`

Extracts visual scene information from sampled video frames using
SmolVLM2.

### `object_tracker.py`

Uses YOLO11 and ByteTrack to detect road users, maintain persistent
track IDs, count objects, store trajectories, and generate an annotated
tracking video.

### `object_analyzer.py`

Analyzes tracked entities to estimate track reliability and motion
characteristics such as approaching or receding movement, apparent
motion, and lateral displacement.

### `interaction_analyzer.py`

Compares trusted tracked entities across event windows to identify
traffic conflicts, vulnerable road-user interactions, and possible
accident events.

### `fuse_qwen.py`

Uses Qwen2.5 to combine scene, motion, tracking, and interaction
evidence into structured JSON output.

## Running the Analysis

Activate the virtual environment:

``` bash
source venv/bin/activate
```

Run object tracking:

``` bash
python object_tracker.py videos/v8.mp4
```

Run object motion analysis:

``` bash
python object_analyzer.py results/v8_tracking.json
```

Run interaction analysis:

``` bash
python interaction_analyzer.py results/v8_tracking.json results/v8_object_analysis.json
```

The generated JSON files and annotated videos are stored in the
`results/` directory.


## Goal

The goal of this project is to move beyond basic video captioning and
build an evidence-driven traffic video understanding system capable of
explaining **what is present, how objects move, which road users
interact, and which interactions may represent safety-critical events**.

## Status

The current pipeline supports object tracking, motion analysis, and
object-to-object interaction analysis. Ego-vehicle interaction reasoning
and final Qwen-based accident explanation are being further developed.
