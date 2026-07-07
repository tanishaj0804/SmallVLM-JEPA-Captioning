# Temporal-Aware Video Understanding using SmolVLM2, V-JEPA 2 and Qwen2.5

## Overview

Vision-Language Models are effective at describing the visual content of a video, including people, objects, vehicles, and the surrounding environment. However, visually similar videos may contain different actions or motion patterns that are difficult to distinguish from scene information alone.

For example, a person **picking up a cup** and a person **putting down a cup** may contain the same person, object, and background while differing mainly in temporal motion.

This project combines **visual scene understanding**, **temporal action understanding**, and **evidence fusion** to produce a structured analysis of a video.

---

## Project Idea

The system uses three models with complementary roles.

### SmolVLM2 — Visual Scene Understanding

SmolVLM2 analyzes representative frames sampled from the input video and generates a description of the visible scene.

It provides visual evidence such as:

- Environment and setting
- People or pedestrians
- Vehicles and objects
- Visible activities
- Background and scene context

Representative video frames are sampled using OpenCV before being passed to SmolVLM2.


### V-JEPA 2 — Temporal Action Understanding

V-JEPA 2 analyzes motion across the video.

The V-JEPA 2 model used in this project includes an action classification probe trained on the Something-Something V2 action dataset. Instead of generating natural-language descriptions, it produces ranked action hypotheses.

The pipeline selects the **top 3 action hypotheses with confidence greater than or equal to 5%**.

These predictions are treated as temporal hypotheses rather than guaranteed simultaneous actions.

---

### Qwen2.5-3B-Instruct — Evidence Fusion

The visual evidence from SmolVLM2 and the selected temporal hypotheses from V-JEPA 2 are passed to Qwen2.5-3B-Instruct.

Qwen does not directly process the original video. Its role is to fuse and ground the outputs of the two video models.

The fusion stage:

- Combines visual and temporal evidence
- Grounds motion hypotheses when supported by visible entities
- Preserves unresolved motion when an action cannot be assigned confidently
- Identifies evidence-supported safety concerns
- Avoids unsupported details where possible
- Produces structured JSON output

Qwen2.5-3B-Instruct is loaded using 4-bit quantization to reduce model memory usage.

---

## System Pipeline

```text
                    Input Video
                         |
          +--------------+--------------+
          |                             |
          v                             v
      SmolVLM2                       V-JEPA 2
  Visual Understanding          Temporal Understanding
          |                             |
          v                             v
   Scene Description          Ranked Action Hypotheses
                                        |
                              Top 3 AND confidence >= 5%
          |                             |
          +--------------+--------------+
                         |
                         v
                Qwen2.5-3B-Instruct
                   Evidence Fusion
                         |
                         v
                Structured JSON Output
```

---

## Output

The final result is stored as a JSON file in the `results/` directory.

Example structure:

```json
{
  "scene_summary": "",
  "people_present": false,
  "people_description": "",
  "objects": [],
  "environment": "",
  "grounded_actions": [],
  "unresolved_motion": [],
  "safety_concerns": [],
  "risk_level": "unknown",
  "notable_details": ""
}
```

The complete pipeline output also stores the original visual evidence, V-JEPA predictions, selected temporal hypotheses, and final structured analysis.

---

## Project Structure

```text
SmallVLM-JEPA-Captioning/
|
├── scene_caption.py
├── action_probe.py
├── fusion_model.py
├── pipeline.py
├── requirements.txt
├── README.md
|
├── videos/
|   └── sample videos
|
└── results/
    └── generated JSON files
```

### `scene_caption.py`

Samples representative frames from the video using OpenCV and uses SmolVLM2 to generate visual scene evidence.

### `action_probe.py`

Uses the V-JEPA 2 action classification probe to generate ranked temporal action hypotheses.

### `fusion_model.py`

Uses Qwen2.5-3B-Instruct to combine visual and temporal evidence and generate grounded structured JSON.

### `pipeline.py`

Runs the complete three-model video understanding pipeline and saves the final result.

---

## Installation

Create and activate a Python virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The models are downloaded from Hugging Face when they are used for the first time.

---

## Running the Project

Run the complete pipeline:

```bash
python pipeline.py videos/v3.mp4
```

The generated result is written to:

```text
results/v3.json
```

The visual and temporal stages can also be tested independently:

```bash
python scene_caption.py videos/v3.mp4 --num_frames 8
```

```bash
python action_probe.py videos/v3.mp4
```

---

## Goal

The goal of this project is to improve video understanding by combining complementary model capabilities:

- **SmolVLM2** provides semantic and visual scene information.
- **V-JEPA 2** provides temporal motion hypotheses.
- **Qwen2.5-3B-Instruct** performs evidence grounding and structured fusion.

Instead of relying on a single model for scene understanding, motion understanding, and reasoning, the project separates these tasks across specialized models and combines their outputs into a structured video analysis.
