from __future__ import annotations

import argparse
import gc
import json
import re

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from PIL import Image

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)


# ============================================================================
# MODEL / CONFIGURATION
# ============================================================================

MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

DEFAULT_CONTEXT_SECONDS = 0.50
DEFAULT_NUM_FRAMES = 6
DEFAULT_FRAME_MAX_SIDE = 320
DEFAULT_MAX_NEW_TOKENS = 12

ROUTED_DECISIONS = {
    "VISUAL_CONFIRMATION_REQUIRED",
    "CRITICAL_CANDIDATE",
}

VALID_ANSWERS = {
    "YES",
    "NO",
    "UNCLEAR",
}


# ============================================================================
# VISUAL QUESTIONS
# ============================================================================

QUESTIONS = {
    "contact": """
These images are sequential frames from one synchronized traffic camera view.
They are ordered from earliest to latest.

Question:
Is visible physical contact between road users shown in this sequence?

Physical contact means the bodies or vehicles visibly touch or strike.
Do not infer contact only because objects appear close, overlap in image space,
pass behind each other, or are temporarily occluded.

Answer with exactly one word:
YES
NO
UNCLEAR
""".strip(),

    "consequence": """
These images are sequential frames from one synchronized traffic camera view.
They are ordered from earliest to latest.

Question:
Does a road user visibly show a physical consequence during this sequence?

A physical consequence includes:
- falling
- abrupt stopping after an interaction
- visible loss of control
- abnormal rotation
- sudden displacement
- a rider becoming separated from a motorcycle or bicycle

Ordinary turning, passing, camera motion, perspective change, or temporary
occlusion are not physical consequences.

Answer with exactly one word:
YES
NO
UNCLEAR
""".strip(),

    "temporal_link": """
These images are sequential frames from one synchronized traffic camera view.
They are ordered from earliest to latest.

Question:
If an interaction and a visible physical consequence are present, does the
physical consequence occur immediately after the interaction?

Answer YES only when the temporal order is visibly:
interaction first, physical consequence immediately after.

Answer NO when the visible motion is ordinary or the consequence is unrelated.
Answer UNCLEAR when the ordering or event is obscured.

Answer with exactly one word:
YES
NO
UNCLEAR
""".strip(),
}


# ============================================================================
# DEVICE
# ============================================================================

def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def get_dtype(
    device: torch.device,
) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16

    return torch.float32


# ============================================================================
# JSON
# ============================================================================

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
            f"JSON root must be a dictionary: {path}"
        )

    return data


# ============================================================================
# VIDEO
# ============================================================================

def get_video_metadata(
    path: Path,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video: {path}"
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
            f"Could not determine FPS: {path}"
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


def load_candidate_frames(
    video_path: Path,
    start_time: float,
    end_time: float,
    num_frames: int,
) -> Tuple[np.ndarray, List[float]]:
    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video: {video_path}"
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
            f"Could not determine FPS: {video_path}"
        )

    duration = (
        total_frames / fps
        if total_frames > 0
        else end_time
    )

    start_time = max(
        0.0,
        float(start_time),
    )

    end_time = min(
        duration,
        float(end_time),
    )

    if end_time <= start_time:
        cap.release()

        raise ValueError(
            "Candidate visual interval is empty."
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

        frames.append(
            frame
        )

        decoded_times.append(
            round(
                frame_index / fps,
                4,
            )
        )

    cap.release()

    if not frames:
        raise RuntimeError(
            "Could not decode any frames "
            f"from {video_path}"
        )

    return (
        np.stack(frames),
        decoded_times,
    )


def resize_frame_for_vlm(
    frame: np.ndarray,
    max_side: int = DEFAULT_FRAME_MAX_SIDE,
) -> Image.Image:
    image = Image.fromarray(
        frame
    )

    width, height = image.size

    largest_side = max(
        width,
        height,
    )

    if largest_side <= max_side:
        return image

    scale = (
        max_side / largest_side
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


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def parse_binary_answer(
    text: str,
) -> str:
    cleaned = (
        text
        .strip()
        .upper()
    )

    exact = re.fullmatch(
        r"\s*(YES|NO|UNCLEAR)\s*[.!]?\s*",
        cleaned,
    )

    if exact is not None:
        return exact.group(1)

    tokens = re.findall(
        r"\b(YES|NO|UNCLEAR)\b",
        cleaned,
    )

    unique_tokens = set(
        tokens
    )

    if len(unique_tokens) == 1:
        return next(
            iter(
                unique_tokens
            )
        )

    return "UNCLEAR"


# ============================================================================
# VLM
# ============================================================================

class PhysicalConsequenceVerifier:

    def __init__(
        self,
    ) -> None:
        self.device = get_device()
        self.dtype = get_dtype(
            self.device
        )

        self.processor = None
        self.model = None

    def load(
        self,
    ) -> None:
        if (
            self.processor is not None
            and self.model is not None
        ):
            return

        print(
            "\n[classifier] Loading visual model..."
        )

        print(
            f"[classifier] Model  : {MODEL_ID}"
        )

        print(
            f"[classifier] Device : {self.device}"
        )

        print(
            f"[classifier] Dtype  : {self.dtype}"
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                MODEL_ID
            )
        )

        self.model = (
            AutoModelForImageTextToText
            .from_pretrained(
                MODEL_ID,
                dtype=self.dtype,
            )
            .to(
                self.device
            )
        )

        self.model.eval()

        print(
            "[classifier] Visual model loaded."
        )

    @torch.inference_mode()
    def ask(
        self,
        frames: np.ndarray,
        question_name: str,
    ) -> Dict[str, str]:
        self.load()

        prompt = QUESTIONS[
            question_name
        ]

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
                "text": prompt,
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
            self.device
        )

        generated_ids = (
            self.model.generate(
                **inputs,
                do_sample=False,
                use_cache=False,
                max_new_tokens=(
                    DEFAULT_MAX_NEW_TOKENS
                ),
            )
        )

        input_length = (
            inputs[
                "input_ids"
            ].shape[1]
        )

        new_tokens = generated_ids[
            :,
            input_length:
        ]

        raw_response = (
            self.processor
            .batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )[0]
            .strip()
        )

        answer = parse_binary_answer(
            raw_response
        )

        result = {
            "answer": answer,
            "raw_response": raw_response,
        }

        # Release per-question multimodal tensors immediately.
        # This is important on 4 GB GPUs because the next question
        # otherwise may need another large contiguous vision allocation.
        del generated_ids
        del new_tokens
        del inputs
        del frame_images
        del content
        del messages

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def unload(
        self,
    ) -> None:
        self.model = None
        self.processor = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# CRITICALITY ROUTING
# ============================================================================

def extract_routed_candidates(
    criticality_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    events = criticality_data.get(
        "criticality_events",
        []
    )

    if not isinstance(
        events,
        list,
    ):
        raise ValueError(
            "criticality_events must be a list."
        )

    routed = []

    for event in events:
        if not isinstance(
            event,
            dict,
        ):
            continue

        decision = str(
            event.get(
                "decision",
                ""
            )
        ).strip().upper()

        if decision not in ROUTED_DECISIONS:
            continue

        if (
            "start_time" not in event
            or "end_time" not in event
        ):
            continue

        routed.append(
            event
        )

    routed.sort(
        key=lambda event: int(
            event.get(
                "rank",
                999999,
            )
        )
    )

    return routed


# ============================================================================
# DETERMINISTIC FUSION
# ============================================================================

def classify_candidate(
    view_results: Dict[
        str,
        Dict[str, str],
    ],
) -> Tuple[str, List[str]]:
    reasons = []

    complete_positive_views = []

    consequence_link_views = []

    for (
        view_name,
        answers,
    ) in view_results.items():
        contact = answers[
            "contact"
        ]

        consequence = answers[
            "consequence"
        ]

        temporal_link = answers[
            "temporal_link"
        ]

        if (
            contact == "YES"
            and consequence == "YES"
            and temporal_link == "YES"
        ):
            complete_positive_views.append(
                view_name
            )

        if (
            consequence == "YES"
            and temporal_link == "YES"
        ):
            consequence_link_views.append(
                view_name
            )

    if complete_positive_views:
        reasons.append(
            "At least one synchronized view shows "
            "contact, physical consequence, and the "
            "correct temporal order."
        )

        reasons.append(
            "Complete positive views: "
            + ", ".join(
                complete_positive_views
            )
        )

        return (
            "YES",
            reasons,
        )

    if (
        len(
            consequence_link_views
        )
        >= 2
    ):
        reasons.append(
            "At least two synchronized views show "
            "a physical consequence immediately "
            "after the interaction."
        )

        reasons.append(
            "Supporting views: "
            + ", ".join(
                consequence_link_views
            )
        )

        return (
            "YES",
            reasons,
        )

    all_consequence_no = all(
        answers[
            "consequence"
        ] == "NO"
        for answers in view_results.values()
    )

    all_contact_no = all(
        answers[
            "contact"
        ] == "NO"
        for answers in view_results.values()
    )

    any_unclear = any(
        answer == "UNCLEAR"
        for answers in view_results.values()
        for answer in answers.values()
    )

    any_consequence_yes = any(
        answers[
            "consequence"
        ] == "YES"
        for answers in view_results.values()
    )

    any_temporal_yes = any(
        answers[
            "temporal_link"
        ] == "YES"
        for answers in view_results.values()
    )

    if (
        all_consequence_no
        and all_contact_no
        and not any_unclear
    ):
        reasons.append(
            "No synchronized view shows contact "
            "or a visible physical consequence."
        )

        return (
            "NO",
            reasons,
        )

    if (
        not any_consequence_yes
        and not any_temporal_yes
        and not any_unclear
    ):
        reasons.append(
            "The synchronized views do not show "
            "a consequence-linked event."
        )

        return (
            "NO",
            reasons,
        )

    reasons.append(
        "Visual evidence is mixed, incomplete, "
        "or obscured across synchronized views."
    )

    return (
        "UNCLEAR",
        reasons,
    )


def classify_case(
    candidate_results: List[
        Dict[str, Any]
    ],
) -> Tuple[str, List[str]]:
    yes_candidates = [
        result
        for result in candidate_results
        if result[
            "candidate_decision"
        ] == "YES"
    ]

    unclear_candidates = [
        result
        for result in candidate_results
        if result[
            "candidate_decision"
        ] == "UNCLEAR"
    ]

    if yes_candidates:
        return (
            "YES",
            [
                (
                    "At least one routed criticality "
                    "candidate was visually confirmed."
                ),
                (
                    "Confirmed candidates: "
                    + ", ".join(
                        str(
                            result[
                                "fused_event_id"
                            ]
                        )
                        for result in yes_candidates
                    )
                ),
            ],
        )

    if unclear_candidates:
        return (
            "UNCLEAR",
            [
                (
                    "No candidate was visually confirmed, "
                    "but at least one candidate retained "
                    "mixed or obscured visual evidence."
                ),
            ],
        )

    return (
        "NO",
        [
            (
                "No routed candidate was visually "
                "confirmed as an accident."
            ),
        ],
    )


# ============================================================================
# LOGGING
# ============================================================================

def print_case_header(
    case_name: str,
    criticality_path: Path,
    views: Dict[str, Path],
    routed: List[Dict[str, Any]],
) -> None:
    print(
        "\n"
        + "=" * 100
    )

    print(
        "MULTIVIEW EVENT CLASSIFIER V4 MEMORY-SAFE"
    )

    print(
        "=" * 100
    )

    print(
        f"Case                    : {case_name}"
    )

    print(
        f"Criticality JSON        : {criticality_path}"
    )

    print(
        f"Available views         : {list(views.keys())}"
    )

    for (
        view_name,
        video_path,
    ) in views.items():
        print(
            f"  {view_name:<20}: {video_path}"
        )

    print(
        f"Routed decisions        : "
        f"{sorted(ROUTED_DECISIONS)}"
    )

    print(
        f"Candidates to verify    : {len(routed)}"
    )

    if routed:
        for event in routed:
            print(
                "  "
                f"{event.get('fused_event_id')} | "
                f"rank={event.get('rank')} | "
                f"decision={event.get('decision')} | "
                f"route={event.get('criticality_route')} | "
                f"window="
                f"{float(event['start_time']):.2f}"
                f"-"
                f"{float(event['end_time']):.2f}s"
            )

    print(
        "=" * 100
    )


def print_question_result(
    question_name: str,
    result: Dict[str, str],
) -> None:
    print(
        f"      question       : "
        f"{question_name.upper()}"
    )

    print(
        f"      raw response   : "
        f"{repr(result['raw_response'])}"
    )

    print(
        f"      parsed answer  : "
        f"{result['answer']}"
    )


# ============================================================================
# COMPLETE CLASSIFIER
# ============================================================================

def run_classifier(
    criticality_json_path: str,
    view_specs: List[str],
    output_path: Optional[str] = None,
    context_seconds: float = (
        DEFAULT_CONTEXT_SECONDS
    ),
    num_frames: int = (
        DEFAULT_NUM_FRAMES
    ),
) -> Dict[str, Any]:
    criticality_path = Path(
        criticality_json_path
    )

    if not criticality_path.is_file():
        raise FileNotFoundError(
            "Criticality JSON not found: "
            f"{criticality_path}"
        )

    views: Dict[
        str,
        Path,
    ] = {}

    for view_spec in view_specs:
        if "=" not in view_spec:
            raise ValueError(
                "--view must use NAME=VIDEO_PATH format. "
                f"Received: {view_spec}"
            )

        view_name, raw_path = (
            view_spec.split(
                "=",
                1,
            )
        )

        view_name = (
            view_name
            .strip()
            .upper()
        )

        video_path = Path(
            raw_path.strip()
        )

        if not view_name:
            raise ValueError(
                "View name cannot be empty."
            )

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video for {view_name} not found: "
                f"{video_path}"
            )

        views[
            view_name
        ] = video_path

    if not views:
        raise ValueError(
            "At least one --view NAME=VIDEO_PATH "
            "must be provided."
        )

    criticality_data = load_json(
        criticality_path
    )

    case_name = str(
        criticality_data.get(
            "case",
            criticality_path.stem,
        )
    )

    routed_candidates = (
        extract_routed_candidates(
            criticality_data
        )
    )

    print_case_header(
        case_name=case_name,
        criticality_path=(
            criticality_path
        ),
        views=views,
        routed=routed_candidates,
    )

    candidate_results = []

    verifier = (
        PhysicalConsequenceVerifier()
    )

    if routed_candidates:
        try:
            verifier.load()

            for (
                candidate_index,
                candidate,
            ) in enumerate(
                routed_candidates,
                start=1,
            ):
                fused_event_id = (
                    candidate.get(
                        "fused_event_id",
                        f"Candidate_{candidate_index}",
                    )
                )

                detected_start = float(
                    candidate[
                        "start_time"
                    ]
                )

                detected_end = float(
                    candidate[
                        "end_time"
                    ]
                )

                print(
                    "\n"
                    + "-" * 100
                )

                print(
                    f"CANDIDATE {candidate_index} : "
                    f"{fused_event_id}"
                )

                print(
                    "-" * 100
                )

                print(
                    f"  source rank       : "
                    f"{candidate.get('rank')}"
                )

                print(
                    f"  criticality input : "
                    f"{candidate.get('decision')}"
                )

                print(
                    f"  criticality route : "
                    f"{candidate.get('criticality_route')}"
                )

                print(
                    f"  detected window   : "
                    f"{detected_start:.2f}"
                    f"-"
                    f"{detected_end:.2f}s"
                )

                print(
                    f"  reasons           : "
                    f"{candidate.get('reasons', [])}"
                )

                view_results = {}
                view_details = {}

                for (
                    view_name,
                    video_path,
                ) in views.items():
                    metadata = (
                        get_video_metadata(
                            video_path
                        )
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

                    print(
                        f"\n  VIEW : {view_name}"
                    )

                    print(
                        f"    video           : "
                        f"{video_path}"
                    )

                    print(
                        f"    fps             : "
                        f"{metadata['fps']:.4f}"
                    )

                    print(
                        f"    total frames    : "
                        f"{metadata['total_frames']}"
                    )

                    print(
                        f"    duration        : "
                        f"{metadata['duration_seconds']:.4f}s"
                    )

                    print(
                        f"    visual window   : "
                        f"{visual_start:.2f}"
                        f"-"
                        f"{visual_end:.2f}s"
                    )

                    frames, frame_times = (
                        load_candidate_frames(
                            video_path=(
                                video_path
                            ),
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

                    print(
                        f"    sampled frames  : "
                        f"{len(frames)}"
                    )

                    print(
                        f"    sampled times   : "
                        f"{frame_times}"
                    )

                    answers = {}
                    raw_responses = {}

                    for question_name in (
                        "contact",
                        "consequence",
                        "temporal_link",
                    ):
                        question_result = (
                            verifier.ask(
                                frames=frames,
                                question_name=(
                                    question_name
                                ),
                            )
                        )

                        answers[
                            question_name
                        ] = question_result[
                            "answer"
                        ]

                        raw_responses[
                            question_name
                        ] = question_result[
                            "raw_response"
                        ]

                        print_question_result(
                            question_name,
                            question_result,
                        )

                    view_results[
                        view_name
                    ] = answers

                    view_details[
                        view_name
                    ] = {
                        "video_path": str(
                            video_path
                        ),
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
                        "answers": answers,
                        "raw_responses": (
                            raw_responses
                        ),
                    }

                    print(
                        f"    VIEW ANSWERS    : "
                        f"CONTACT={answers['contact']} | "
                        f"CONSEQUENCE={answers['consequence']} | "
                        f"TEMPORAL_LINK={answers['temporal_link']}"
                    )

                (
                    candidate_decision,
                    fusion_reasons,
                ) = classify_candidate(
                    view_results
                )

                print(
                    "\n  DETERMINISTIC MULTIVIEW FUSION"
                )

                print(
                    f"    view answers     : "
                    f"{view_results}"
                )

                print(
                    f"    fusion reasons   : "
                    f"{fusion_reasons}"
                )

                print(
                    f"    candidate result : "
                    f"{candidate_decision}"
                )

                candidate_results.append(
                    {
                        "candidate_index": (
                            candidate_index
                        ),
                        "fused_event_id": (
                            fused_event_id
                        ),
                        "source_rank": (
                            candidate.get(
                                "rank"
                            )
                        ),
                        "criticality_decision": (
                            candidate.get(
                                "decision"
                            )
                        ),
                        "criticality_route": (
                            candidate.get(
                                "criticality_route"
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
                        "views": view_details,
                        "candidate_decision": (
                            candidate_decision
                        ),
                        "fusion_reasons": (
                            fusion_reasons
                        ),
                    }
                )

        finally:
            verifier.unload()

    else:
        print(
            "\n[classifier] No event was routed for "
            "visual confirmation."
        )

        print(
            "[classifier] Zero VLM calls are required."
        )

    (
        final_decision,
        final_reasons,
    ) = classify_case(
        candidate_results
    )

    output = {
        "configuration": {
            "version": "v4-memory-safe",
            "model_id": MODEL_ID,
            "input_stage": (
                "criticality_analyzer"
            ),
            "routed_decisions": sorted(
                ROUTED_DECISIONS
            ),
            "verification_questions": [
                "contact",
                "consequence",
                "temporal_link",
            ],
            "context_seconds": (
                context_seconds
            ),
            "num_frames_per_view": (
                num_frames
            ),
            "final_label_space": [
                "YES",
                "NO",
                "UNCLEAR",
            ],
        },
        "case": case_name,
        "source_criticality_json": str(
            criticality_path
        ),
        "views": {
            view_name: str(
                video_path
            )
            for (
                view_name,
                video_path,
            ) in views.items()
        },
        "routed_candidate_count": len(
            routed_candidates
        ),
        "candidate_classifications": (
            candidate_results
        ),
        "final_classification": {
            "accident": final_decision,
            "reasons": final_reasons,
        },
    }

    if output_path is None:
        output_file = (
            criticality_path.parent
            / (
                criticality_path.stem
                .replace(
                    "_criticality_analysis",
                    "",
                )
                + "_event_classification.json"
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

    print(
        "\n"
        + "=" * 100
    )

    print(
        "EVENT CLASSIFIER COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"Case                    : "
        f"{case_name}"
    )

    print(
        f"Routed candidates       : "
        f"{len(routed_candidates)}"
    )

    print(
        f"Candidates classified   : "
        f"{len(candidate_results)}"
    )

    for result in candidate_results:
        print(
            f"  {result['fused_event_id']:<20} "
            f"{result['candidate_decision']}"
        )

    print(
        f"Final fusion reasons    : "
        f"{final_reasons}"
    )

    print(
        f"Saved to                : "
        f"{output_file}"
    )

    print(
        "=" * 100
    )

    # ------------------------------------------------------------------------
    # PROFESSOR-FRIENDLY FINAL OUTPUT
    # The LAST printed line is intentionally only YES / NO / UNCLEAR.
    # ------------------------------------------------------------------------

    print(
        final_decision
    )

    return output


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a multiview traffic event from "
            "criticality-routed candidates using three "
            "physical-consequence questions per view."
        )
    )

    parser.add_argument(
        "criticality_json",
        help=(
            "Path to *_criticality_analysis.json"
        ),
    )

    parser.add_argument(
        "--view",
        action="append",
        required=True,
        help=(
            "Synchronized camera video in "
            "NAME=VIDEO_PATH format. Repeat once "
            "for every camera view."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON path."
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

    args = parser.parse_args()

    run_classifier(
        criticality_json_path=(
            args.criticality_json
        ),
        view_specs=(
            args.view
        ),
        output_path=(
            args.output
        ),
        context_seconds=(
            args.context_seconds
        ),
        num_frames=(
            args.num_frames
        ),
    )


if __name__ == "__main__":
    main()
