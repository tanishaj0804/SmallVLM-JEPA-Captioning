from __future__ import annotations

import argparse
import gc
import json
import re

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from PIL import Image

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)


# ============================================================
# MODEL
# ============================================================

MODEL_ID = (
    "HuggingFaceTB/"
    "SmolVLM2-256M-Video-Instruct"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MAX_CANDIDATES = 3

DEFAULT_CONTEXT_SECONDS = 0.5

DEFAULT_NUM_FRAMES = 6

DEFAULT_FRAME_MAX_SIDE = 384

DEFAULT_MAX_NEW_TOKENS = 220

DEFAULT_MAX_TEMPORAL_IOU = 0.35


# ============================================================
# VERIFICATION PROMPT
# ============================================================

VERIFICATION_PROMPT = """
The images are sequential frames from one short traffic video segment.
They are ordered from earliest to latest.

Compare the first frames with the last frames.

Decide whether a clearly visible critical traffic event occurs.

A CRITICAL event requires a visible consequence such as:
- two road users visibly colliding
- a rider or pedestrian visibly falling
- a vehicle visibly losing control
- a road user making a sudden evasive movement to avoid an immediate collision
- a sudden obstruction visibly causing another road user to react

NORMAL includes:
- ordinary driving
- approaching vehicles
- vehicles passing each other
- turning
- perspective changes
- camera movement
- temporary occlusion
- vehicles merely appearing close in the image

Answer using exactly four lines.

DECISION: CRITICAL or NORMAL
EVENT: collision, fall, loss_of_control, near_collision, obstruction, or none
CHANGE: one short sentence describing the visible change from early to late frames
EVIDENCE: one short sentence describing the strongest visible evidence

Do not describe the general scene.
Do not describe each image separately.
Do not mention lighting, roads, trees, or weather unless directly involved.
Do not output JSON.
""".strip()


# ============================================================
# DEVICE
# ============================================================

def _device() -> torch.device:

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def _dtype(
    device: torch.device,
) -> torch.dtype:

    if device.type == "cuda":
        return torch.float16

    return torch.float32


# ============================================================
# JSON LOADING
# ============================================================

def load_json(
    path: Path,
) -> Dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):

        raise ValueError(
            "Temporal analysis JSON "
            "must contain a dictionary."
        )

    return data


# ============================================================
# TEMPORAL WINDOWS
# ============================================================

def extract_ranked_windows(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:

    windows = data.get(
        "ranked_windows"
    )

    if not isinstance(windows, list):

        raise ValueError(
            "Could not find ranked_windows "
            "in temporal analysis JSON."
        )

    valid_windows = []

    for window in windows:

        if not isinstance(window, dict):
            continue

        if (
            "start_time" not in window
            or "end_time" not in window
        ):
            continue

        valid_windows.append(window)

    if not valid_windows:

        raise ValueError(
            "No valid temporal candidate "
            "windows were found."
        )

    return sorted(
        valid_windows,
        key=lambda window: int(
            window.get(
                "rank",
                10**9,
            )
        ),
    )


# ============================================================
# TEMPORAL IOU
# ============================================================

def temporal_iou(
    first: Dict[str, Any],
    second: Dict[str, Any],
) -> float:

    first_start = float(
        first["start_time"]
    )

    first_end = float(
        first["end_time"]
    )

    second_start = float(
        second["start_time"]
    )

    second_end = float(
        second["end_time"]
    )

    intersection = max(
        0.0,
        min(
            first_end,
            second_end,
        )
        - max(
            first_start,
            second_start,
        ),
    )

    union = (
        max(
            first_end,
            second_end,
        )
        - min(
            first_start,
            second_start,
        )
    )

    if union <= 0.0:
        return 0.0

    return intersection / union


# ============================================================
# TEMPORAL NON-MAXIMUM SUPPRESSION
# ============================================================

def select_temporally_diverse_candidates(
    ranked_windows: List[
        Dict[str, Any]
    ],
    max_candidates: int,
    max_temporal_iou: float,
) -> List[Dict[str, Any]]:

    selected: List[
        Dict[str, Any]
    ] = []

    for window in ranked_windows:

        overlaps_existing = False

        for existing in selected:

            overlap = temporal_iou(
                window,
                existing,
            )

            if (
                overlap
                > max_temporal_iou
            ):

                overlaps_existing = True

                break

        if overlaps_existing:
            continue

        selected.append(window)

        if (
            len(selected)
            >= max_candidates
        ):
            break

    return selected


# ============================================================
# VIDEO METADATA
# ============================================================

def get_video_metadata(
    path: Path,
) -> Dict[str, float]:

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"OpenCV could not open "
            f"video: {path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    if fps <= 0.0:

        raise RuntimeError(
            f"Could not determine FPS: "
            f"{path}"
        )

    duration = (
        total_frames / fps
        if total_frames > 0
        else 0.0
    )

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration,
    }


# ============================================================
# FRAME EXTRACTION
# ============================================================

def load_candidate_frames(
    video_path: Path,
    start_time: float,
    end_time: float,
    num_frames: int,
) -> tuple[
    np.ndarray,
    List[float],
]:

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"OpenCV could not open "
            f"video: {video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0.0:

        cap.release()

        raise RuntimeError(
            "Could not determine "
            "video FPS."
        )

    duration = (
        total_frames / fps
        if total_frames > 0
        else end_time
    )

    start_time = max(
        0.0,
        start_time,
    )

    end_time = min(
        duration,
        end_time,
    )

    if end_time <= start_time:

        cap.release()

        raise ValueError(
            "Candidate temporal interval "
            "is empty."
        )

    sample_times = np.linspace(
        start_time,
        end_time,
        num=num_frames,
        dtype=np.float64,
    )

    frames = []

    decoded_times = []

    for sample_time in sample_times:

        frame_index = int(
            round(
                sample_time * fps
            )
        )

        if total_frames > 0:

            frame_index = min(
                frame_index,
                total_frames - 1,
            )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)

        decoded_times.append(
            round(
                frame_index / fps,
                4,
            )
        )

    cap.release()

    if not frames:

        raise RuntimeError(
            "Could not decode frames "
            "for candidate interval."
        )

    return (
        np.stack(frames),
        decoded_times,
    )


# ============================================================
# FRAME RESIZING
# ============================================================

def resize_frame_for_vlm(
    frame: np.ndarray,
    max_side: int = (
        DEFAULT_FRAME_MAX_SIDE
    ),
) -> Image.Image:

    image = Image.fromarray(frame)

    width, height = image.size

    largest_side = max(
        width,
        height,
    )

    if largest_side <= max_side:

        return image

    scale = (
        max_side
        / float(largest_side)
    )

    new_width = max(
        1,
        int(
            round(
                width * scale
            )
        ),
    )

    new_height = max(
        1,
        int(
            round(
                height * scale
            )
        ),
    )

    return image.resize(
        (
            new_width,
            new_height,
        ),
        Image.Resampling.BILINEAR,
    )


# ============================================================
# VISUAL DECISION PARSING
# ============================================================

def parse_visual_decision(
    text: str,
) -> Dict[str, Any]:

    cleaned = text.strip()

    decision_match = re.search(
        r"DECISION\s*:\s*(CRITICAL|NORMAL)",
        cleaned,
        flags=re.IGNORECASE,
    )

    event_match = re.search(
        (
            r"EVENT\s*:\s*"
            r"(collision|fall|loss_of_control|"
            r"near_collision|obstruction|none)"
        ),
        cleaned,
        flags=re.IGNORECASE,
    )

    change_match = re.search(
        r"CHANGE\s*:\s*(.+)",
        cleaned,
        flags=re.IGNORECASE,
    )

    evidence_match = re.search(
        r"EVIDENCE\s*:\s*(.+)",
        cleaned,
        flags=re.IGNORECASE,
    )

    if decision_match is None:
        raise ValueError(
            "VLM response did not contain "
            "a DECISION field."
        )

    decision = (
        decision_match
        .group(1)
        .strip()
        .upper()
    )

    critical_event = (
        decision == "CRITICAL"
    )

    if event_match is None:
        event_type = (
            "unspecified"
            if critical_event
            else "none"
        )
    else:
        event_type = (
            event_match
            .group(1)
            .strip()
            .lower()
        )

    if not critical_event:
        event_type = "none"

    temporal_change = (
        change_match.group(1).strip()
        if change_match
        else ""
    )

    evidence = (
        evidence_match.group(1).strip()
        if evidence_match
        else ""
    )

    visible_evidence = (
        [evidence]
        if evidence
        else []
    )

    return {
        "critical_event": critical_event,
        "event_type": event_type,
        "involved_road_users": [],
        "temporal_change": temporal_change,
        "visible_evidence": visible_evidence,
        "uncertainty": "none",
        # This is deliberately neutral: the four-line response is a
        # categorical decision, not a calibrated probability.
        "confidence": 0.5,
    }


# ============================================================
# RESULT VALIDATION
# ============================================================

# ============================================================

def validate_verification(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    critical_event = result.get(
        "critical_event",
        False,
    )

    if not isinstance(
        critical_event,
        bool,
    ):

        critical_event = (
            str(
                critical_event
            )
            .strip()
            .lower()
            == "true"
        )

    event_type = str(
        result.get(
            "event_type",
            "none",
        )
    ).strip()

    involved = result.get(
        "involved_road_users",
        [],
    )

    if not isinstance(
        involved,
        list,
    ):

        involved = []

    involved = [
        str(item).strip()
        for item in involved
        if str(item).strip()
    ]

    temporal_change = str(
        result.get(
            "temporal_change",
            "",
        )
    ).strip()

    visible_evidence = result.get(
        "visible_evidence",
        [],
    )

    if not isinstance(
        visible_evidence,
        list,
    ):

        visible_evidence = []

    visible_evidence = [
        str(item).strip()
        for item in visible_evidence
        if str(item).strip()
    ]

    uncertainty = str(
        result.get(
            "uncertainty",
            "none",
        )
    ).strip()

    try:

        confidence = float(
            result.get(
                "confidence",
                0.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        confidence = 0.0

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

    if not critical_event:

        event_type = "none"

    return {
        "critical_event": (
            critical_event
        ),
        "event_type": event_type,
        "involved_road_users": involved,
        "temporal_change": (
            temporal_change
        ),
        "visible_evidence": (
            visible_evidence
        ),
        "uncertainty": uncertainty,
        "confidence": round(
            confidence,
            4,
        ),
    }



# ============================================================
# SMOLVLM2 VERIFIER
# ============================================================

class EventVerifier:

    def __init__(
        self,
        model_id: str = MODEL_ID,
    ) -> None:

        self.model_id = model_id
        self.device = _device()
        self.dtype = _dtype(
            self.device
        )
        self.processor = None
        self.model = None


    def load(
        self,
    ) -> "EventVerifier":

        if self.model is not None:
            return self

        print(
            f"[verify] Loading "
            f"{self.model_id} "
            f"on {self.device}..."
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                self.model_id
            )
        )

        self.model = (
            AutoModelForImageTextToText
            .from_pretrained(
                self.model_id,
                dtype=self.dtype,
            )
            .to(self.device)
        )

        self.model.eval()

        return self


    @torch.inference_mode()
    def verify_frames(
        self,
        frames: np.ndarray,
        max_new_tokens: int = 120,
    ) -> tuple[
        Dict[str, Any],
        str,
    ]:

        self.load()

        frame_images = [
            resize_frame_for_vlm(
                frame
            )
            for frame in frames
        ]

        content = []

        for image in frame_images:
            content.append(
                {
                    "type": "image",
                    "image": image,
                }
            )

        content.append(
            {
                "type": "text",
                "text": VERIFICATION_PROMPT,
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = (
            self.processor
            .apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        inputs = inputs.to(
            self.device,
            dtype=self.dtype,
        )

        print(
            f"  input tokens     : "
            f"{inputs['input_ids'].shape[1]}"
        )

        print(
            f"  visual frames    : "
            f"{len(frame_images)}"
        )

        frame_sizes = [
            image.size
            for image in frame_images
        ]

        print(
            f"  frame sizes      : "
            f"{frame_sizes}"
        )

        generated_ids = (
            self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        )

        input_length = (
            inputs["input_ids"].shape[1]
        )

        new_tokens = generated_ids[
            :,
            input_length:
        ]

        text = (
            self.processor
            .batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )[0]
            .strip()
        )

        print(
            "  raw VLM response : "
            f"{repr(text[:500])}"
        )

        try:
            parsed = parse_visual_decision(
                text
            )

            print(
                "  decision parse   : SUCCESS"
            )

        except ValueError as error:
            print(
                "  decision parse   : FAILED"
            )

            print(
                f"  parse reason     : {error}"
            )

            parsed = {
                "critical_event": False,
                "event_type": "none",
                "involved_road_users": [],
                "temporal_change": "",
                "visible_evidence": [],
                "uncertainty": (
                    "VLM decision output "
                    "could not be parsed."
                ),
                "confidence": 0.0,
            }

        validated = validate_verification(
            parsed
        )

        return (
            validated,
            text,
        )


    def unload(
        self,
    ) -> None:

        self.model = None
        self.processor = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# FINAL EVENT SELECTION
# ============================================================

# ============================================================

def choose_final_event(
    candidates: List[
        Dict[str, Any]
    ],
) -> Optional[Dict[str, Any]]:

    positive_candidates = [
        candidate
        for candidate in candidates
        if candidate[
            "verification"
        ][
            "critical_event"
        ]
    ]

    if not positive_candidates:

        return None

    return max(
        positive_candidates,
        key=lambda candidate: (
            candidate[
                "verification"
            ][
                "confidence"
            ],
            candidate[
                "temporal_abnormality_score"
            ],
        ),
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def verify_temporal_candidates(
    video_path: str,
    temporal_json_path: str,
    output_path: Optional[str] = None,
    max_candidates: int = (
        DEFAULT_MAX_CANDIDATES
    ),
    context_seconds: float = (
        DEFAULT_CONTEXT_SECONDS
    ),
    num_frames: int = (
        DEFAULT_NUM_FRAMES
    ),
    max_temporal_iou: float = (
        DEFAULT_MAX_TEMPORAL_IOU
    ),
) -> Dict[str, Any]:

    video = Path(
        video_path
    )

    temporal_path = Path(
        temporal_json_path
    )

    if not video.is_file():

        raise FileNotFoundError(
            f"Video not found: "
            f"{video}"
        )

    if not temporal_path.is_file():

        raise FileNotFoundError(
            "Temporal analysis JSON "
            f"not found: "
            f"{temporal_path}"
        )

    print(
        "[verify] Loading temporal "
        "analysis..."
    )

    temporal_data = load_json(
        temporal_path
    )

    ranked_windows = (
        extract_ranked_windows(
            temporal_data
        )
    )

    selected_windows = (
        select_temporally_diverse_candidates(
            ranked_windows,
            max_candidates=(
                max_candidates
            ),
            max_temporal_iou=(
                max_temporal_iou
            ),
        )
    )

    metadata = get_video_metadata(
        video
    )

    print(
        f"[verify] Selected "
        f"{len(selected_windows)} "
        f"temporally diverse candidates."
    )

    verifier = EventVerifier()

    candidate_results = []

    try:

        verifier.load()

        for (
            candidate_index,
            window,
        ) in enumerate(
            selected_windows,
            start=1,
        ):

            detected_start = float(
                window["start_time"]
            )

            detected_end = float(
                window["end_time"]
            )

            visual_start = max(
                0.0,
                detected_start
                - context_seconds,
            )

            visual_end = min(
                metadata[
                    "duration_seconds"
                ],
                detected_end
                + context_seconds,
            )

            print()

            print(
                f"[verify] Candidate "
                f"{candidate_index}"
            )

            print(
                f"  detected window : "
                f"{detected_start:.2f}"
                f" - "
                f"{detected_end:.2f} s"
            )

            print(
                f"  visual window   : "
                f"{visual_start:.2f}"
                f" - "
                f"{visual_end:.2f} s"
            )

            frames, frame_times = (
                load_candidate_frames(
                    video,
                    start_time=(
                        visual_start
                    ),
                    end_time=(
                        visual_end
                    ),
                    num_frames=(
                        num_frames
                    ),
                )
            )

            (
                verification,
                raw_response,
            ) = verifier.verify_frames(
                frames
            )

            print(
                f"  critical event  : "
                f"{verification['critical_event']}"
            )

            print(
                f"  event type      : "
                f"{verification['event_type']}"
            )

            print(
                f"  confidence      : "
                f"{verification['confidence']:.4f}"
            )

            candidate_result = {
                "candidate_index": (
                    candidate_index
                ),
                "source_window_id": (
                    window.get(
                        "window_id"
                    )
                ),
                "source_rank": (
                    window.get(
                        "rank"
                    )
                ),
                "detected_window": {
                    "start_time": (
                        detected_start
                    ),
                    "end_time": (
                        detected_end
                    ),
                },
                "visual_window": {
                    "start_time": round(
                        visual_start,
                        4,
                    ),
                    "end_time": round(
                        visual_end,
                        4,
                    ),
                },
                "sampled_frame_times": (
                    frame_times
                ),
                "temporal_state_score": (
                    window.get(
                        "state_score"
                    )
                ),
                "temporal_transition_score": (
                    window.get(
                        "transition_score"
                    )
                ),
                "temporal_abnormality_score": (
                    float(
                        window.get(
                            "abnormality_score",
                            0.0,
                        )
                    )
                ),
                "temporal_evidence": (
                    window.get(
                        "evidence",
                        {},
                    )
                ),
                "transition_evidence": (
                    window.get(
                        "transition_evidence",
                        {},
                    )
                ),
                "verification": (
                    verification
                ),
                "raw_vlm_response": (
                    raw_response
                ),
            }

            candidate_results.append(
                candidate_result
            )

    finally:

        verifier.unload()

    final_candidate = choose_final_event(
        candidate_results
    )

    if final_candidate is None:

        final_event = {
            "critical_event_detected": False,
            "event_start_seconds": None,
            "event_end_seconds": None,
            "event_type": "none",
            "confidence": 0.0,
            "involved_road_users": [],
            "temporal_change": "",
            "visible_evidence": [],
            "uncertainty": (
                "No candidate was visually "
                "verified as a critical event."
            ),
        }

    else:

        verification = final_candidate[
            "verification"
        ]

        visual_window = final_candidate[
            "visual_window"
        ]

        final_event = {
            "critical_event_detected": True,
            "event_start_seconds": (
                visual_window[
                    "start_time"
                ]
            ),
            "event_end_seconds": (
                visual_window[
                    "end_time"
                ]
            ),
            "event_type": (
                verification[
                    "event_type"
                ]
            ),
            "confidence": (
                verification[
                    "confidence"
                ]
            ),
            "involved_road_users": (
                verification[
                    "involved_road_users"
                ]
            ),
            "temporal_change": (
                verification[
                    "temporal_change"
                ]
            ),
            "visible_evidence": (
                verification[
                    "visible_evidence"
                ]
            ),
            "uncertainty": (
                verification[
                    "uncertainty"
                ]
            ),
            "selected_candidate_index": (
                final_candidate[
                    "candidate_index"
                ]
            ),
        }

    output = {
        "configuration": {
            "model_id": MODEL_ID,
            "candidate_selection": (
                "temporal_non_maximum_suppression"
            ),
            "max_candidates": (
                max_candidates
            ),
            "max_temporal_iou": (
                max_temporal_iou
            ),
            "context_seconds": (
                context_seconds
            ),
            "num_frames_per_candidate": (
                num_frames
            ),
            "frame_max_side": (
                DEFAULT_FRAME_MAX_SIDE
            ),
            "max_new_tokens": (
                DEFAULT_MAX_NEW_TOKENS
            ),
            "semantic_verification": True,
            "accident_specific_thresholds": False,
        },
        "video": {
            "path": str(
                video
            ),
            "fps": metadata[
                "fps"
            ],
            "total_frames": metadata[
                "total_frames"
            ],
            "duration_seconds": metadata[
                "duration_seconds"
            ],
        },
        "candidate_verifications": (
            candidate_results
        ),
        "final_event": (
            final_event
        ),
    }

    if output_path is None:

        output_file = (
            temporal_path.parent
            / (
                temporal_path.stem
                .replace(
                    "_temporal_analysis",
                    "",
                )
                + "_event_verification.json"
            )
        )

    else:

        output_file = Path(
            output_path
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print()

    print(
        "=" * 88
    )

    print(
        "EVENT VERIFIER COMPLETE"
    )

    print(
        "=" * 88
    )

    print(
        f"Candidates verified : "
        f"{len(candidate_results)}"
    )

    print(
        f"Critical event       : "
        f"{final_event['critical_event_detected']}"
    )

    print(
        f"Event type           : "
        f"{final_event['event_type']}"
    )

    print(
        f"Confidence           : "
        f"{final_event['confidence']:.4f}"
    )

    print(
        f"Saved to             : "
        f"{output_file}"
    )

    return output


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Verify temporally suspicious "
            "traffic events using SmolVLM2."
        )
    )

    parser.add_argument(
        "video",
        help="Path to original video.",
    )

    parser.add_argument(
        "temporal_json",
        help=(
            "Path to temporal analysis JSON."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path.",
    )

    parser.add_argument(
        "--max_candidates",
        type=int,
        default=(
            DEFAULT_MAX_CANDIDATES
        ),
    )

    parser.add_argument(
        "--context_seconds",
        type=float,
        default=(
            DEFAULT_CONTEXT_SECONDS
        ),
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=(
            DEFAULT_NUM_FRAMES
        ),
    )

    parser.add_argument(
        "--max_temporal_iou",
        type=float,
        default=(
            DEFAULT_MAX_TEMPORAL_IOU
        ),
    )

    args = parser.parse_args()

    verify_temporal_candidates(
        video_path=(
            args.video
        ),
        temporal_json_path=(
            args.temporal_json
        ),
        output_path=(
            args.output
        ),
        max_candidates=(
            args.max_candidates
        ),
        context_seconds=(
            args.context_seconds
        ),
        num_frames=(
            args.num_frames
        ),
        max_temporal_iou=(
            args.max_temporal_iou
        ),
    )


if __name__ == "__main__":

    main()
