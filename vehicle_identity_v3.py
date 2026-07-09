"""
Track-level vehicle visual identity using CLIP.

Identifies:
    - vehicle color
    - vehicle subtype

Examples:
    White_Sedan
    Black_SUV
    Red_Motorcycle
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch

from PIL import Image

from transformers import (
    CLIPModel,
    CLIPProcessor,
)


CLIP_MODEL_NAME = (
    "openai/clip-vit-base-patch32"
)


SUPPORTED_TYPES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}


COLOR_LABELS = [
    "black",
    "white",
    "silver",
    "gray",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
]


SUBTYPE_LABELS = {
    "car": [
        "sedan",
        "SUV",
        "hatchback",
        "van",
    ],

    "truck": [
        "pickup truck",
        "cargo truck",
        "semi truck",
    ],

    "bus": [
        "bus",
        "minibus",
    ],

    "motorcycle": [
        "motorcycle",
        "scooter",
    ],

    "bicycle": [
        "bicycle",
    ],
}


NUM_REPRESENTATIVE_FRAMES = 12

MIN_CROP_WIDTH = 40
MIN_CROP_HEIGHT = 40

MIN_COLOR_TRACK_CONFIDENCE = 0.45
MIN_SUBTYPE_TRACK_CONFIDENCE = 0.45

MIN_SAMPLE_PROBABILITY = 0.20


def select_representative_detections(
    detections,
    num_samples=NUM_REPRESENTATIVE_FRAMES,
):
    valid_detections = [
        detection
        for detection in detections
        if (
            detection["bbox"][2]
            - detection["bbox"][0]
            >= MIN_CROP_WIDTH
            and detection["bbox"][3]
            - detection["bbox"][1]
            >= MIN_CROP_HEIGHT
        )
    ]

    if not valid_detections:
        return []

    valid_detections = sorted(
        valid_detections,
        key=lambda detection: (
            detection["area_ratio"],
            detection["confidence"],
        ),
        reverse=True,
    )

    candidate_count = min(
        len(valid_detections),
        num_samples * 3,
    )

    candidates = valid_detections[
        :candidate_count
    ]

    candidates = sorted(
        candidates,
        key=lambda detection: (
            detection["frame"]
        ),
    )

    if len(candidates) <= num_samples:
        return candidates

    indices = np.linspace(
        0,
        len(candidates) - 1,
        num_samples,
        dtype=int,
    )

    return [
        candidates[index]
        for index in indices
    ]


def extract_crop(
    frame,
    bbox,
):
    frame_height, frame_width = (
        frame.shape[:2]
    )

    x1, y1, x2, y2 = [
        int(value)
        for value in bbox
    ]

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(
        frame_width,
        x2,
    )

    y2 = min(
        frame_height,
        y2,
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[
        y1:y2,
        x1:x2,
    ]


class VisualIdentityClassifier:
    def __init__(self):
        print(
            "[identity] Loading CLIP..."
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = (
            CLIPModel.from_pretrained(
                CLIP_MODEL_NAME
            )
        )

        self.processor = (
            CLIPProcessor.from_pretrained(
                CLIP_MODEL_NAME
            )
        )

        self.model.to(self.device)

        self.model.eval()

        print(
            f"[identity] CLIP device: "
            f"{self.device}"
        )


    def run_clip(
        self,
        crop,
        prompts,
        labels,
    ):
        rgb_crop = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            rgb_crop
        )

        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value
            in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(
                **inputs
            )

            probabilities = (
                outputs
                .logits_per_image
                .softmax(dim=1)
                .cpu()
                .numpy()[0]
            )

        best_index = int(
            np.argmax(probabilities)
        )

        return (
            labels[best_index],
            float(
                probabilities[best_index]
            ),
            {
                label: round(
                    float(probability),
                    4,
                )
                for label, probability
                in zip(
                    labels,
                    probabilities,
                )
            },
        )


    def classify_color(
        self,
        crop,
        object_type,
    ):
        prompts = [
            (
                f"a photo of a "
                f"{color} {object_type}"
            )
            for color in COLOR_LABELS
        ]

        return self.run_clip(
            crop,
            prompts,
            COLOR_LABELS,
        )


    def classify_subtype(
        self,
        crop,
        object_type,
    ):
        labels = SUBTYPE_LABELS.get(
            object_type
        )

        if not labels:
            return (
                object_type,
                0.0,
                {},
            )

        prompts = [
            (
                f"a traffic camera photo "
                f"of a {label}"
            )
            for label in labels
        ]

        return self.run_clip(
            crop,
            prompts,
            labels,
        )


def calculate_track_result(
    weighted_votes,
    fallback,
    minimum_confidence,
):
    if not weighted_votes:
        return (
            fallback,
            0.0,
        )

    best_label, best_score = max(
        weighted_votes.items(),
        key=lambda item: item[1],
    )

    total_score = sum(
        weighted_votes.values()
    )

    if total_score <= 0:
        return (
            fallback,
            0.0,
        )

    confidence = (
        best_score / total_score
    )

    if confidence < minimum_confidence:
        return (
            fallback,
            confidence,
        )

    return (
        best_label,
        confidence,
    )


def analyze_track(
    video_path,
    detections,
    object_type,
    classifier,
):
    selected_detections = (
        select_representative_detections(
            detections
        )
    )

    color_votes = defaultdict(float)

    subtype_votes = defaultdict(float)

    samples = []

    cap = cv2.VideoCapture(
        video_path
    )

    for detection in selected_detections:
        frame_number = detection["frame"]

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number,
        )

        success, frame = cap.read()

        if not success:
            continue

        crop = extract_crop(
            frame,
            detection["bbox"],
        )

        if crop is None:
            continue

        (
            color,
            color_probability,
            color_probabilities,
        ) = classifier.classify_color(
            crop,
            object_type,
        )

        (
            subtype,
            subtype_probability,
            subtype_probabilities,
        ) = classifier.classify_subtype(
            crop,
            object_type,
        )

        if (
            color_probability
            >= MIN_SAMPLE_PROBABILITY
        ):
            color_votes[color] += (
                color_probability
            )

        if (
            subtype_probability
            >= MIN_SAMPLE_PROBABILITY
        ):
            subtype_votes[subtype] += (
                subtype_probability
            )

        samples.append(
            {
                "frame": frame_number,
                "color": color,
                "color_probability": round(
                    color_probability,
                    4,
                ),
                "color_probabilities": (
                    color_probabilities
                ),
                "subtype": subtype,
                "subtype_probability": round(
                    subtype_probability,
                    4,
                ),
                "subtype_probabilities": (
                    subtype_probabilities
                ),
            }
        )

    cap.release()

    (
        stable_color,
        color_confidence,
    ) = calculate_track_result(
        color_votes,
        "unknown",
        MIN_COLOR_TRACK_CONFIDENCE,
    )

    (
        stable_subtype,
        subtype_confidence,
    ) = calculate_track_result(
        subtype_votes,
        object_type,
        MIN_SUBTYPE_TRACK_CONFIDENCE,
    )

    return {
        "color": stable_color,
        "color_confidence": round(
            color_confidence,
            4,
        ),
        "color_votes": {
            key: round(value, 4)
            for key, value
            in color_votes.items()
        },
        "subtype": stable_subtype,
        "subtype_confidence": round(
            subtype_confidence,
            4,
        ),
        "subtype_votes": {
            key: round(value, 4)
            for key, value
            in subtype_votes.items()
        },
        "samples": samples,
    }


def build_visual_name(
    color,
    subtype,
):
    subtype_name = (
        subtype
        .replace(" ", "_")
        .title()
    )

    if color == "unknown":
        return subtype_name

    return (
        f"{color.capitalize()}_"
        f"{subtype_name}"
    )


def make_unique_names(
    identity_results,
):
    name_counts = Counter(
        result["base_visual_name"]
        for result
        in identity_results.values()
    )

    current_counts = Counter()

    for result in (
        identity_results.values()
    ):
        base_name = result[
            "base_visual_name"
        ]

        if name_counts[base_name] == 1:
            result["visual_name"] = (
                base_name
            )

            continue

        current_counts[base_name] += 1

        result["visual_name"] = (
            f"{base_name}_"
            f"{current_counts[base_name]}"
        )


def analyze_tracking_file(
    video_path,
    tracking_json,
    output_json,
):
    with open(
        tracking_json,
        "r",
        encoding="utf-8",
    ) as file:
        tracking_data = json.load(file)

    entities = {
        entity["id"]: entity
        for entity
        in tracking_data[
            "tracked_entities"
        ]
    }

    trajectories = tracking_data[
        "trajectories"
    ]

    classifier = (
        VisualIdentityClassifier()
    )

    identity_results = {}

    for entity_id, detections in (
        trajectories.items()
    ):
        entity = entities.get(entity_id)

        if entity is None:
            continue

        object_type = entity["type"]

        if (
            object_type
            not in SUPPORTED_TYPES
        ):
            continue

        print(
            f"\n[identity] Analyzing "
            f"{entity_id}..."
        )

        result = analyze_track(
            video_path,
            detections,
            object_type,
            classifier,
        )

        base_visual_name = (
            build_visual_name(
                result["color"],
                result["subtype"],
            )
        )

        identity_results[entity_id] = {
            "original_id": entity_id,
            "type": object_type,
            "base_visual_name": (
                base_visual_name
            ),
            **result,
        }

        print(
            f"  Color   : "
            f"{result['color']} "
            f"(track confidence="
            f"{result['color_confidence']:.3f})"
        )

        print(
            f"  Subtype : "
            f"{result['subtype']} "
            f"(track confidence="
            f"{result['subtype_confidence']:.3f})"
        )

    make_unique_names(
        identity_results
    )

    print(
        "\n"
        + "=" * 72
    )

    print("VISUAL IDENTITIES")

    print("=" * 72)

    for (
        entity_id,
        result,
    ) in identity_results.items():
        print(
            f"{entity_id:<20} -> "
            f"{result['visual_name']}"
        )

    output = {
        "video_path": video_path,
        "tracking_json": tracking_json,
        "identity_mode": (
            "clip_color_and_subtype"
        ),
        "identities": identity_results,
    }

    os.makedirs(
        os.path.dirname(
            output_json
        ) or ".",
        exist_ok=True,
    )

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"\nIdentity JSON written to: "
        f"{output_json}"
    )

    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "CLIP track-level vehicle "
            "visual identity"
        )
    )

    parser.add_argument(
        "video",
        help="Input video path",
    )

    parser.add_argument(
        "tracking_json",
        help="Tracking JSON path",
    )

    parser.add_argument(
        "--output_json",
        default=None,
        help="Output identity JSON",
    )

    args = parser.parse_args()

    video_name = os.path.splitext(
        os.path.basename(args.video)
    )[0]

    output_json = (
        args.output_json
        or (
            f"results/"
            f"{video_name}_identity.json"
        )
    )

    analyze_tracking_file(
        args.video,
        args.tracking_json,
        output_json,
    )


if __name__ == "__main__":
    main()
