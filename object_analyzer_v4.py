"""
object_analyzer.py

Temporal road-object motion state analyzer.

Responsibilities:
    - estimate camera-induced scene motion
    - compensate object trajectories for scene motion
    - calculate causal temporal motion states
    - estimate image-space depth trends
    - estimate motion persistence and stability
    - produce synchronized object states
    - smooth scene-flow estimates over time
    - calibrate motion confidence using trajectory stability
    - estimate conservative ego-corridor intrusion evidence
    - emit causal motion-transition events

This file DOES NOT classify accidents.

IMPORTANT:
2D bounding-box overlap is NOT treated as collision evidence.

Pairwise conflict reasoning belongs in:
    interaction_analyzer.py
"""

from __future__ import annotations

import argparse
import json
import math

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

TIMELINE_STEP_SECONDS = 0.10

MOTION_WINDOW_SECONDS = 0.90

MIN_MOTION_POINTS = 4

MAX_POINT_STALENESS_SECONDS = 0.20


# --------------------------------------------------------------------------
# SCENE FLOW
# --------------------------------------------------------------------------

VANISHING_X = 0.50
VANISHING_Y = 0.38

MIN_SCENE_ENTITIES = 5

MAD_SCALE = 3.5

SCENE_FLOW_SMOOTHING_SECONDS = 0.70
SCENE_FLOW_MIN_CONFIDENCE = 0.20
SCENE_FLOW_ENTITY_TARGET = 10
SCENE_FLOW_RESIDUAL_SCALE = 0.12
SCENE_FLOW_TEMPORAL_SCALE = 0.35


# --------------------------------------------------------------------------
# MOTION CLASSIFICATION
# --------------------------------------------------------------------------

STATIONARY_SPEED = 0.018

LATERAL_SPEED = 0.030

VERTICAL_SPEED = 0.020

AREA_RATE_THRESHOLD = 0.025


# --------------------------------------------------------------------------
# APPROACH / DEPTH
# --------------------------------------------------------------------------

APPROACH_VY_THRESHOLD = 0.018

APPROACH_AREA_THRESHOLD = 0.025

RECEDE_VY_THRESHOLD = -0.018

RECEDE_AREA_THRESHOLD = -0.025


# --------------------------------------------------------------------------
# RELIABILITY
# --------------------------------------------------------------------------

MIN_TRACK_DURATION = 0.30

MIN_MOTION_CONFIDENCE = 0.25

VELOCITY_STABILITY_SCALE = 0.08

ACCELERATION_SCALE = 0.20

LOW_STABILITY_THRESHOLD = 0.15
MODERATE_STABILITY_THRESHOLD = 0.30
LOW_STABILITY_CONFIDENCE_FACTOR = 0.35
MODERATE_STABILITY_CONFIDENCE_FACTOR = 0.60


# --------------------------------------------------------------------------
# EGO CORRIDOR
# --------------------------------------------------------------------------

EGO_CORRIDOR_TOP_Y = 0.38
EGO_CORRIDOR_BOTTOM_Y = 1.00
EGO_CORRIDOR_TOP_HALF_WIDTH = 0.07
EGO_CORRIDOR_BOTTOM_HALF_WIDTH = 0.22

EGO_CURRENT_OVERLAP_THRESHOLD = 0.18
EGO_TOWARD_MIN_LATERAL_SPEED = 0.018
EGO_INTRUSION_RATE_SCALE = 0.08


# --------------------------------------------------------------------------
# MOTION EVENTS
# --------------------------------------------------------------------------

EVENT_MIN_CONFIDENCE = 0.25
RAPID_SPEED_THRESHOLD = 0.065
LOW_SPEED_THRESHOLD = 0.022
EVENT_LATERAL_SPEED_THRESHOLD = 0.035

# Compare the current state against a short causal history instead of only
# the immediately preceding 0.10 s state. This preserves fast-to-slow and
# slow-to-fast transitions that unfold over several timeline samples.
EVENT_LOOKBACK_SECONDS = 0.80
EVENT_MIN_HISTORY_STATES = 3

DEPTH_CONFIRMATION_STATES = 3
EVENT_COOLDOWN_SECONDS = 0.50

SPEED_TRANSITION_WINDOW_SECONDS = 0.30
SPEED_DROP_RATIO_THRESHOLD = 0.58
SPEED_RISE_RATIO_THRESHOLD = 1.75
RELATIVE_LOW_SPEED_FRACTION = 0.45
MIN_REFERENCE_SPEED = 0.020


# ============================================================================
# UTILITIES
# ============================================================================


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:

    return max(
        low,
        min(high, value),
    )


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def bbox_center(
    bbox: List[float],
) -> Tuple[float, float]:

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def bbox_width(
    bbox: List[float],
) -> float:

    return max(
        0.0,
        bbox[2] - bbox[0],
    )


def bbox_height(
    bbox: List[float],
) -> float:

    return max(
        0.0,
        bbox[3] - bbox[1],
    )


def bbox_area(
    bbox: List[float],
) -> float:

    return (
        bbox_width(bbox)
        * bbox_height(bbox)
    )



def bbox_intersection_area(
    box_a: List[float],
    box_b: List[float],
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def linear_slope(
    times: List[float],
    values: List[float],
) -> float:

    if len(times) < 2:
        return 0.0

    x = np.asarray(
        times,
        dtype=np.float64,
    )

    y = np.asarray(
        values,
        dtype=np.float64,
    )

    if np.std(x) < 1e-9:
        return 0.0

    return float(
        np.polyfit(
            x,
            y,
            1,
        )[0]
    )


def robust_median(
    values: List[float],
) -> float:

    if not values:
        return 0.0

    return float(
        np.median(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


# ============================================================================
# INPUT
# ============================================================================


def get_metadata(
    data: Dict[str, Any],
) -> Tuple[float, float, float]:

    metadata = data.get(
        "metadata",
        data.get(
            "video_metadata",
            {},
        ),
    )

    fps = safe_float(
        metadata.get(
            "fps",
            30.0,
        ),
        30.0,
    )

    width = safe_float(
        metadata.get(
            "frame_width",
            metadata.get(
                "width",
                1.0,
            ),
        ),
        1.0,
    )

    height = safe_float(
        metadata.get(
            "frame_height",
            metadata.get(
                "height",
                1.0,
            ),
        ),
        1.0,
    )

    return (
        fps,
        width,
        height,
    )


def get_entity_metadata(
    data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:

    containers = (
        "tracked_entities",
        "physical_entities",
        "entities",
        "tracks",
        "objects",
    )

    for key in containers:

        value = data.get(key)

        if isinstance(value, list):

            result = {}

            for item in value:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                entity_id = (
                    item.get("entity_id")
                    or item.get("track_id")
                    or item.get("object_id")
                    or item.get("id")
                    or item.get("name")
                )

                if entity_id is not None:

                    result[
                        str(entity_id)
                    ] = item

            if result:
                return result

        if isinstance(value, dict):

            result = {}

            for (
                entity_id,
                item,
            ) in value.items():

                if isinstance(
                    item,
                    dict,
                ):

                    result[
                        str(entity_id)
                    ] = item

            if result:
                return result

    return {}


def get_class_name(
    metadata: Dict[str, Any],
) -> str:

    for key in (
        "class_name",
        "class",
        "label",
        "object_class",
        "category",
        "type",
    ):

        value = metadata.get(key)

        if value is not None:

            return str(value)

    return "object"


def normalize_bbox(
    bbox: List[float],
    width: float,
    height: float,
) -> List[float]:

    bbox = [
        safe_float(value)
        for value in bbox
    ]

    if max(
        abs(value)
        for value in bbox
    ) > 2.0:

        return [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height,
        ]

    return bbox


def extract_trajectories(
    data: Dict[str, Any],
) -> Dict[
    str,
    List[Dict[str, Any]],
]:

    fps, width, height = get_metadata(
        data
    )

    raw_trajectories = data.get(
        "trajectories",
        {},
    )

    if not isinstance(
        raw_trajectories,
        dict,
    ):

        raise ValueError(
            "Tracking JSON does not contain "
            "top-level trajectories."
        )

    trajectories = {}

    for (
        entity_id,
        raw_points,
    ) in raw_trajectories.items():

        if not isinstance(
            raw_points,
            list,
        ):
            continue

        points = []

        for (
            index,
            item,
        ) in enumerate(raw_points):

            if not isinstance(
                item,
                dict,
            ):
                continue

            bbox = (
                item.get("bbox")
                or item.get("box")
                or item.get("xyxy")
                or item.get(
                    "bounding_box"
                )
            )

            if (
                bbox is None
                or len(bbox) != 4
            ):
                continue

            bbox = normalize_bbox(
                bbox,
                width,
                height,
            )

            frame = int(
                item.get(
                    "frame",
                    item.get(
                        "frame_index",
                        index,
                    ),
                )
            )

            timestamp = safe_float(
                item.get(
                    "timestamp",
                    item.get(
                        "time",
                        item.get(
                            "time_seconds",
                            frame / fps,
                        ),
                    ),
                )
            )

            center_x, center_y = (
                bbox_center(bbox)
            )

            points.append(
                {
                    "frame": frame,
                    "timestamp": timestamp,
                    "bbox": bbox,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": bbox_width(
                        bbox
                    ),
                    "height": bbox_height(
                        bbox
                    ),
                    "area": bbox_area(
                        bbox
                    ),
                }
            )

        points.sort(
            key=lambda point: (
                point["timestamp"],
                point["frame"],
            )
        )

        if points:

            trajectories[
                str(entity_id)
            ] = points

    return trajectories


def load_identity_map(
    identity_path: Optional[Path],
) -> Dict[str, str]:

    if (
        identity_path is None
        or not identity_path.exists()
    ):
        return {}

    with identity_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    candidates = data.get(
        "identities",
        data.get(
            "vehicle_identities",
            data,
        ),
    )

    result = {}

    if isinstance(
        candidates,
        dict,
    ):

        for (
            entity_id,
            value,
        ) in candidates.items():

            if isinstance(
                value,
                str,
            ):

                result[
                    str(entity_id)
                ] = value

            elif isinstance(
                value,
                dict,
            ):

                identity = (
                    value.get(
                        "visual_identity"
                    )
                    or value.get("identity")
                    or value.get("label")
                    or value.get("name")
                )

                if identity:

                    result[
                        str(entity_id)
                    ] = str(identity)

    elif isinstance(
        candidates,
        list,
    ):

        for item in candidates:

            if not isinstance(
                item,
                dict,
            ):
                continue

            entity_id = (
                item.get("entity_id")
                or item.get("track_id")
                or item.get("id")
            )

            identity = (
                item.get(
                    "visual_identity"
                )
                or item.get("identity")
                or item.get("label")
                or item.get("name")
            )

            if (
                entity_id is not None
                and identity
            ):

                result[
                    str(entity_id)
                ] = str(identity)

    return result


# ============================================================================
# MOTION WINDOW
# ============================================================================


def get_causal_window(
    trajectory: List[Dict[str, Any]],
    current_time: float,
) -> List[Dict[str, Any]]:

    start_time = (
        current_time
        - MOTION_WINDOW_SECONDS
    )

    return [
        point
        for point in trajectory
        if (
            start_time
            <= point["timestamp"]
            <= current_time
        )
    ]


# ============================================================================
# RAW MOTION
# ============================================================================


def calculate_raw_motion(
    points: List[Dict[str, Any]],
) -> Dict[str, Any]:

    if len(points) < MIN_MOTION_POINTS:

        return {
            "vx": 0.0,
            "vy": 0.0,
            "width_rate": 0.0,
            "height_rate": 0.0,
            "area_rate": 0.0,
            "reliable": False,
        }

    times = [
        point["timestamp"]
        for point in points
    ]

    center_x = [
        point["center_x"]
        for point in points
    ]

    center_y = [
        point["center_y"]
        for point in points
    ]

    log_width = [
        math.log(
            max(
                point["width"],
                1e-8,
            )
        )
        for point in points
    ]

    log_height = [
        math.log(
            max(
                point["height"],
                1e-8,
            )
        )
        for point in points
    ]

    log_area = [
        math.log(
            max(
                point["area"],
                1e-8,
            )
        )
        for point in points
    ]

    return {
        "vx": linear_slope(
            times,
            center_x,
        ),
        "vy": linear_slope(
            times,
            center_y,
        ),
        "width_rate": linear_slope(
            times,
            log_width,
        ),
        "height_rate": linear_slope(
            times,
            log_height,
        ),
        "area_rate": linear_slope(
            times,
            log_area,
        ),
        "reliable": True,
    }


# ============================================================================
# SCENE FLOW
# ============================================================================


def mad_mask(
    values: np.ndarray,
) -> np.ndarray:

    if len(values) < 4:

        return np.ones(
            len(values),
            dtype=bool,
        )

    center = np.median(values)

    mad = np.median(
        np.abs(
            values - center
        )
    )

    if mad < 1e-9:

        return np.ones(
            len(values),
            dtype=bool,
        )

    sigma = 1.4826 * mad

    return (
        np.abs(
            values - center
        )
        <= MAD_SCALE * sigma
    )


def robust_linear_fit(
    feature: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:

    design = np.column_stack(
        [
            np.ones(
                len(feature)
            ),
            feature,
        ]
    )

    coefficients, _, _, _ = (
        np.linalg.lstsq(
            design,
            target,
            rcond=None,
        )
    )

    residuals = (
        target
        - design @ coefficients
    )

    mask = mad_mask(
        residuals
    )

    if np.sum(mask) >= 2:

        coefficients, _, _, _ = (
            np.linalg.lstsq(
                design[mask],
                target[mask],
                rcond=None,
            )
        )

    return (
        float(coefficients[0]),
        float(coefficients[1]),
    )


def estimate_scene_flow(
    active_entities: Dict[
        str,
        Dict[str, Any],
    ],
) -> Dict[str, Any]:

    samples = []

    for state in active_entities.values():

        motion = state["raw_motion"]

        if not motion["reliable"]:
            continue

        point = state["latest"]

        samples.append(
            {
                "x": point["center_x"],
                "y": point["center_y"],
                "vx": motion["vx"],
                "vy": motion["vy"],
                "area_rate": motion["area_rate"],
            }
        )

    empty = {
        "lateral_intercept": 0.0,
        "lateral_radial": 0.0,
        "vertical_intercept": 0.0,
        "vertical_radial": 0.0,
        "area_intercept": 0.0,
        "area_vertical": 0.0,
        "sample_count": len(samples),
        "reliable": False,
        "confidence": 0.0,
        "residual_agreement": 0.0,
    }

    if len(samples) < MIN_SCENE_ENTITIES:
        return empty

    x = np.asarray(
        [sample["x"] - VANISHING_X for sample in samples],
        dtype=np.float64,
    )

    y = np.asarray(
        [sample["y"] - VANISHING_Y for sample in samples],
        dtype=np.float64,
    )

    vx = np.asarray(
        [sample["vx"] for sample in samples],
        dtype=np.float64,
    )

    vy = np.asarray(
        [sample["vy"] for sample in samples],
        dtype=np.float64,
    )

    area_rate = np.asarray(
        [sample["area_rate"] for sample in samples],
        dtype=np.float64,
    )

    lateral_intercept, lateral_radial = robust_linear_fit(x, vx)
    vertical_intercept, vertical_radial = robust_linear_fit(y, vy)
    area_intercept, area_vertical = robust_linear_fit(y, area_rate)

    predicted_vx = lateral_intercept + lateral_radial * x
    predicted_vy = vertical_intercept + vertical_radial * y
    predicted_area = area_intercept + area_vertical * y

    residual = np.sqrt(
        (vx - predicted_vx) ** 2
        + (vy - predicted_vy) ** 2
        + 0.25 * (area_rate - predicted_area) ** 2
    )

    median_residual = float(np.median(residual))
    residual_agreement = clamp(
        1.0 - median_residual / SCENE_FLOW_RESIDUAL_SCALE
    )

    support_score = clamp(
        len(samples) / SCENE_FLOW_ENTITY_TARGET
    )

    confidence = clamp(
        0.45 * support_score
        + 0.55 * residual_agreement
    )

    return {
        "lateral_intercept": lateral_intercept,
        "lateral_radial": lateral_radial,
        "vertical_intercept": vertical_intercept,
        "vertical_radial": vertical_radial,
        "area_intercept": area_intercept,
        "area_vertical": area_vertical,
        "sample_count": len(samples),
        "reliable": confidence >= SCENE_FLOW_MIN_CONFIDENCE,
        "confidence": confidence,
        "residual_agreement": residual_agreement,
    }


SCENE_FLOW_KEYS = (
    "lateral_intercept",
    "lateral_radial",
    "vertical_intercept",
    "vertical_radial",
    "area_intercept",
    "area_vertical",
)


def smooth_scene_flow(
    history: List[Dict[str, Any]],
    raw_scene_flow: Dict[str, Any],
    current_time: float,
) -> Dict[str, Any]:

    candidate = dict(raw_scene_flow)
    candidate["time_seconds"] = current_time

    usable = [
        item
        for item in history
        if (
            item.get("reliable", False)
            and current_time - item["time_seconds"]
            <= SCENE_FLOW_SMOOTHING_SECONDS
        )
    ]

    if raw_scene_flow["reliable"]:
        usable.append(candidate)

    if not usable:
        return raw_scene_flow

    smoothed = {}

    for key in SCENE_FLOW_KEYS:
        smoothed[key] = robust_median(
            [safe_float(item.get(key)) for item in usable]
        )

    current_vector = np.asarray(
        [safe_float(raw_scene_flow.get(key)) for key in SCENE_FLOW_KEYS],
        dtype=np.float64,
    )

    smooth_vector = np.asarray(
        [smoothed[key] for key in SCENE_FLOW_KEYS],
        dtype=np.float64,
    )

    temporal_difference = float(
        np.mean(np.abs(current_vector - smooth_vector))
    )

    temporal_consistency = clamp(
        1.0
        - temporal_difference / SCENE_FLOW_TEMPORAL_SCALE
    )

    base_confidence = safe_float(
        raw_scene_flow.get("confidence")
    )

    if not raw_scene_flow["reliable"]:
        base_confidence = robust_median(
            [safe_float(item.get("confidence")) for item in usable]
        )

    confidence = clamp(
        0.70 * base_confidence
        + 0.30 * temporal_consistency
    )

    previous_authoritative = (
        history[-1]
        if history
        and history[-1].get("reliable", False)
        else None
    )

    discontinuity_blocked = (
        raw_scene_flow["reliable"]
        and previous_authoritative is not None
        and temporal_consistency < 0.15
    )

    if discontinuity_blocked:
        for key in SCENE_FLOW_KEYS:
            smoothed[key] = safe_float(
                previous_authoritative.get(key)
            )

        confidence = min(
            confidence,
            SCENE_FLOW_MIN_CONFIDENCE - 1e-6,
        )

    smoothed.update(
        {
            "sample_count": raw_scene_flow["sample_count"],
            "reliable": (
                confidence >= SCENE_FLOW_MIN_CONFIDENCE
                and not discontinuity_blocked
            ),
            "confidence": confidence,
            "residual_agreement": safe_float(
                raw_scene_flow.get("residual_agreement")
            ),
            "temporal_consistency": temporal_consistency,
            "discontinuity_blocked": discontinuity_blocked,
            "raw_parameters": {
                key: safe_float(raw_scene_flow.get(key))
                for key in SCENE_FLOW_KEYS
            },
        }
    )

    return smoothed


def expected_scene_motion(
    point: Dict[str, Any],
    scene_flow: Dict[str, Any],
) -> Dict[str, float]:

    if not scene_flow["reliable"]:

        return {
            "vx": 0.0,
            "vy": 0.0,
            "area_rate": 0.0,
        }

    x = (
        point["center_x"]
        - VANISHING_X
    )

    y = (
        point["center_y"]
        - VANISHING_Y
    )

    return {
        "vx": (
            scene_flow[
                "lateral_intercept"
            ]
            + scene_flow[
                "lateral_radial"
            ] * x
        ),
        "vy": (
            scene_flow[
                "vertical_intercept"
            ]
            + scene_flow[
                "vertical_radial"
            ] * y
        ),
        "area_rate": (
            scene_flow[
                "area_intercept"
            ]
            + scene_flow[
                "area_vertical"
            ] * y
        ),
    }


# ============================================================================
# TEMPORAL MOTION QUALITY
# ============================================================================


def calculate_velocity_samples(
    points: List[Dict[str, Any]],
) -> Tuple[
    List[float],
    List[float],
]:

    vx_samples = []
    vy_samples = []

    for index in range(
        1,
        len(points),
    ):

        previous = points[
            index - 1
        ]

        current = points[index]

        delta_time = (
            current["timestamp"]
            - previous["timestamp"]
        )

        if delta_time <= 1e-6:
            continue

        vx_samples.append(
            (
                current["center_x"]
                - previous["center_x"]
            )
            / delta_time
        )

        vy_samples.append(
            (
                current["center_y"]
                - previous["center_y"]
            )
            / delta_time
        )

    return (
        vx_samples,
        vy_samples,
    )


def calculate_motion_quality(
    points: List[Dict[str, Any]],
    relative_vx: float,
    relative_vy: float,
) -> Dict[str, Any]:

    if len(points) < MIN_MOTION_POINTS:
        return {
            "duration": 0.0,
            "velocity_stability": 0.0,
            "direction_persistence": 0.0,
            "motion_confidence": 0.0,
            "reliability": "low",
        }

    duration = max(
        points[-1]["timestamp"] - points[0]["timestamp"],
        0.0,
    )

    vx_samples, vy_samples = calculate_velocity_samples(points)

    if not vx_samples:
        return {
            "duration": duration,
            "velocity_stability": 0.0,
            "direction_persistence": 0.0,
            "motion_confidence": 0.0,
            "reliability": "low",
        }

    vx_array = np.asarray(vx_samples, dtype=np.float64)
    vy_array = np.asarray(vy_samples, dtype=np.float64)

    velocity_spread = float(
        np.median(
            np.sqrt(
                (vx_array - np.median(vx_array)) ** 2
                + (vy_array - np.median(vy_array)) ** 2
            )
        )
    )

    velocity_stability = clamp(
        1.0 - velocity_spread / VELOCITY_STABILITY_SCALE
    )

    dominant_speed = math.hypot(relative_vx, relative_vy)

    if dominant_speed < STATIONARY_SPEED:
        direction_persistence = velocity_stability

    else:
        dominant_angle = math.atan2(relative_vy, relative_vx)
        aligned = 0
        total = 0

        for sample_vx, sample_vy in zip(vx_samples, vy_samples):
            speed = math.hypot(sample_vx, sample_vy)

            if speed < 1e-6:
                continue

            angle = math.atan2(sample_vy, sample_vx)

            angle_difference = abs(
                math.atan2(
                    math.sin(angle - dominant_angle),
                    math.cos(angle - dominant_angle),
                )
            )

            if angle_difference <= math.radians(45.0):
                aligned += 1

            total += 1

        direction_persistence = (
            aligned / total
            if total > 0
            else 0.0
        )

    duration_score = clamp(duration / MOTION_WINDOW_SECONDS)
    point_score = clamp(len(points) / 8.0)

    motion_confidence = clamp(
        0.30 * duration_score
        + 0.25 * point_score
        + 0.25 * velocity_stability
        + 0.20 * direction_persistence
    )

    if velocity_stability < LOW_STABILITY_THRESHOLD:
        motion_confidence *= LOW_STABILITY_CONFIDENCE_FACTOR
        reliability = "low"

    elif velocity_stability < MODERATE_STABILITY_THRESHOLD:
        motion_confidence *= MODERATE_STABILITY_CONFIDENCE_FACTOR
        reliability = "moderate"

    elif motion_confidence < 0.55:
        reliability = "moderate"

    else:
        reliability = "high"

    motion_confidence = clamp(motion_confidence)

    return {
        "duration": duration,
        "velocity_stability": velocity_stability,
        "direction_persistence": direction_persistence,
        "motion_confidence": motion_confidence,
        "reliability": reliability,
    }


# ============================================================================
# ACCELERATION
# ============================================================================


def calculate_acceleration(
    points: List[Dict[str, Any]],
) -> Dict[str, float]:

    if len(points) < 6:

        return {
            "ax": 0.0,
            "ay": 0.0,
            "magnitude": 0.0,
        }

    midpoint = len(points) // 2

    first_half = points[:midpoint + 1]

    second_half = points[midpoint:]

    first_motion = (
        calculate_raw_motion(
            first_half
        )
    )

    second_motion = (
        calculate_raw_motion(
            second_half
        )
    )

    first_time = robust_median(
        [
            point["timestamp"]
            for point in first_half
        ]
    )

    second_time = robust_median(
        [
            point["timestamp"]
            for point in second_half
        ]
    )

    delta_time = max(
        second_time - first_time,
        1e-6,
    )

    ax = (
        second_motion["vx"]
        - first_motion["vx"]
    ) / delta_time

    ay = (
        second_motion["vy"]
        - first_motion["vy"]
    ) / delta_time

    return {
        "ax": ax,
        "ay": ay,
        "magnitude": math.hypot(
            ax,
            ay,
        ),
    }


# ============================================================================
# MOTION STATE
# ============================================================================


def classify_depth_state(
    relative_vy: float,
    relative_area_rate: float,
) -> Tuple[str, float]:

    approach_evidence = []

    recede_evidence = []

    approach_evidence.append(
        clamp(
            relative_vy
            / 0.08
        )
    )

    approach_evidence.append(
        clamp(
            relative_area_rate
            / 0.15
        )
    )

    recede_evidence.append(
        clamp(
            -relative_vy
            / 0.08
        )
    )

    recede_evidence.append(
        clamp(
            -relative_area_rate
            / 0.15
        )
    )

    approach_score = (
        0.45
        * approach_evidence[0]
        + 0.55
        * approach_evidence[1]
    )

    recede_score = (
        0.45
        * recede_evidence[0]
        + 0.55
        * recede_evidence[1]
    )

    if (
        relative_vy
        >= APPROACH_VY_THRESHOLD
        or relative_area_rate
        >= APPROACH_AREA_THRESHOLD
    ):

        if approach_score > recede_score:

            return (
                "approaching",
                approach_score,
            )

    if (
        relative_vy
        <= RECEDE_VY_THRESHOLD
        or relative_area_rate
        <= RECEDE_AREA_THRESHOLD
    ):

        if recede_score > approach_score:

            return (
                "receding",
                recede_score,
            )

    confidence = clamp(
        1.0
        - max(
            approach_score,
            recede_score,
        )
    )

    return (
        "stable",
        confidence,
    )


def classify_motion_state(
    relative_vx: float,
    relative_vy: float,
    relative_area_rate: float,
) -> str:

    speed = math.hypot(
        relative_vx,
        relative_vy,
    )

    depth_state, _ = (
        classify_depth_state(
            relative_vy,
            relative_area_rate,
        )
    )

    if (
        speed < STATIONARY_SPEED
        and abs(
            relative_area_rate
        ) < AREA_RATE_THRESHOLD
    ):

        return "stationary"

    horizontal = None

    if (
        relative_vx
        <= -LATERAL_SPEED
    ):

        horizontal = "left"

    elif (
        relative_vx
        >= LATERAL_SPEED
    ):

        horizontal = "right"

    if depth_state == "approaching":

        if horizontal is not None:

            return (
                f"approaching_{horizontal}"
            )

        return "approaching"

    if depth_state == "receding":

        if horizontal is not None:

            return (
                f"receding_{horizontal}"
            )

        return "receding"

    if horizontal is not None:

        return (
            f"moving_{horizontal}"
        )

    if (
        relative_vy
        >= VERTICAL_SPEED
    ):

        return "moving_down"

    if (
        relative_vy
        <= -VERTICAL_SPEED
    ):

        return "moving_up"

    return "slow_motion"


# ============================================================================
# POSITION
# ============================================================================


def classify_position(
    center_x: float,
    center_y: float,
) -> str:

    if center_x < 0.35:
        horizontal = "left"

    elif center_x > 0.65:
        horizontal = "right"

    else:
        horizontal = "center"

    if center_y >= 0.68:
        vertical = "near"

    else:
        vertical = "front"

    if horizontal == "center":

        return vertical

    return (
        f"{vertical}_{horizontal}"
    )


# ============================================================================
# EGO-PATH EVIDENCE
# ============================================================================


def ego_corridor_half_width(
    center_y: float,
) -> float:

    normalized_y = clamp(
        (
            center_y - EGO_CORRIDOR_TOP_Y
        )
        / max(
            EGO_CORRIDOR_BOTTOM_Y - EGO_CORRIDOR_TOP_Y,
            1e-6,
        )
    )

    return (
        EGO_CORRIDOR_TOP_HALF_WIDTH
        + normalized_y
        * (
            EGO_CORRIDOR_BOTTOM_HALF_WIDTH
            - EGO_CORRIDOR_TOP_HALF_WIDTH
        )
    )


def calculate_ego_path_evidence(
    bbox: List[float],
    relative_vx: float,
    motion_confidence: float,
) -> Dict[str, Any]:

    center_x, center_y = bbox_center(bbox)

    half_width = ego_corridor_half_width(center_y)

    corridor_left = 0.5 - half_width
    corridor_right = 0.5 + half_width

    corridor_box = [
        corridor_left,
        bbox[1],
        corridor_right,
        bbox[3],
    ]

    object_area = bbox_area(bbox)

    overlap_ratio = (
        bbox_intersection_area(bbox, corridor_box)
        / max(object_area, 1e-8)
    )

    currently_inside = (
        overlap_ratio >= EGO_CURRENT_OVERLAP_THRESHOLD
    )

    if bbox[2] < corridor_left:
        lateral_distance = corridor_left - bbox[2]
        moving_toward = relative_vx >= EGO_TOWARD_MIN_LATERAL_SPEED
        intrusion_rate = max(relative_vx, 0.0)

    elif bbox[0] > corridor_right:
        lateral_distance = bbox[0] - corridor_right
        moving_toward = relative_vx <= -EGO_TOWARD_MIN_LATERAL_SPEED
        intrusion_rate = max(-relative_vx, 0.0)

    else:
        lateral_distance = 0.0
        moving_toward = False
        intrusion_rate = 0.0

    confidence = clamp(
        motion_confidence
        * (
            0.55
            + 0.45
            * clamp(
                intrusion_rate / EGO_INTRUSION_RATE_SCALE
            )
        )
    )

    return {
        "currently_inside": currently_inside,
        "overlap_ratio": overlap_ratio,
        "lateral_distance_to_corridor": lateral_distance,
        "moving_toward_corridor": moving_toward,
        "intrusion_rate": intrusion_rate,
        "confidence": confidence,
    }


# ============================================================================
# MOTION TRANSITION EVENTS
# ============================================================================


def detect_motion_events(
    previous_states: List[Dict[str, Any]],
    motion_state: str,
    depth_state: str,
    relative_vx: float,
    relative_vy: float,
    relative_area_rate: float,
    motion_confidence: float,
    current_time: float,
) -> List[Dict[str, Any]]:

    if motion_confidence < EVENT_MIN_CONFIDENCE:
        return []

    causal_history = [
        state
        for state in previous_states
        if (
            0.0
            < current_time - state["time_seconds"]
            <= EVENT_LOOKBACK_SECONDS
        )
    ]

    if len(causal_history) < EVENT_MIN_HISTORY_STATES:
        return []

    events = []
    current_speed = math.hypot(relative_vx, relative_vy)

    def recently_emitted(
        event_name: str,
    ) -> bool:

        for state in reversed(previous_states):
            age = current_time - state["time_seconds"]

            if age > EVENT_COOLDOWN_SECONDS:
                break

            for event in state.get("motion_events", []):
                if event.get("event") == event_name:
                    return True

        return False

    def add_event(
        name: str,
        confidence: float,
    ) -> None:

        if recently_emitted(name):
            return

        events.append(
            {
                "time_seconds": round(current_time, 3),
                "event": name,
                "confidence": round(
                    clamp(confidence),
                    4,
                ),
            }
        )

    # --------------------------------------------------------------
    # ENTITY-RELATIVE SPEED TRANSITIONS
    #
    # Compare two causal short windows. This avoids assuming that a
    # distant motorcycle and a nearby car share the same normalized
    # image-space "rapid" threshold.
    # --------------------------------------------------------------

    recent_history = [
        state
        for state in causal_history
        if (
            current_time - state["time_seconds"]
            <= SPEED_TRANSITION_WINDOW_SECONDS
        )
    ]

    earlier_history = [
        state
        for state in causal_history
        if (
            SPEED_TRANSITION_WINDOW_SECONDS
            < current_time - state["time_seconds"]
            <= 2.0 * SPEED_TRANSITION_WINDOW_SECONDS
        )
    ]

    if (
        len(recent_history) >= 2
        and len(earlier_history) >= 2
    ):

        recent_speeds = [
            state["relative_motion"]["speed"]
            for state in recent_history
        ]
        recent_speeds.append(current_speed)

        earlier_speeds = [
            state["relative_motion"]["speed"]
            for state in earlier_history
        ]

        recent_speed = robust_median(recent_speeds)
        earlier_speed = robust_median(earlier_speeds)

        speed_ratio = (
            recent_speed / max(earlier_speed, 1e-8)
        )

        entity_reference_speed = max(
            [
                state["relative_motion"]["speed"]
                for state in causal_history
            ]
            + [current_speed]
        )

        sustained_relative_low = (
            recent_speed
            <= RELATIVE_LOW_SPEED_FRACTION
            * max(entity_reference_speed, 1e-8)
        )

        if (
            earlier_speed >= MIN_REFERENCE_SPEED
            and speed_ratio <= SPEED_DROP_RATIO_THRESHOLD
            and sustained_relative_low
        ):
            transition_strength = clamp(
                1.0 - speed_ratio
            )

            add_event(
                "rapid_to_low_motion",
                motion_confidence
                * (
                    0.65
                    + 0.35 * transition_strength
                ),
            )

        elif (
            recent_speed >= MIN_REFERENCE_SPEED
            and speed_ratio >= SPEED_RISE_RATIO_THRESHOLD
        ):
            transition_strength = clamp(
                (
                    speed_ratio
                    - SPEED_RISE_RATIO_THRESHOLD
                )
                / SPEED_RISE_RATIO_THRESHOLD
            )

            add_event(
                "low_to_rapid_motion",
                motion_confidence
                * (
                    0.65
                    + 0.35 * transition_strength
                ),
            )

    # --------------------------------------------------------------
    # LATERAL MOTION ONSET
    # --------------------------------------------------------------

    historical_vx = [
        state["relative_motion"]["vx"]
        for state in causal_history
    ]

    if (
        max(abs(value) for value in historical_vx)
        < EVENT_LATERAL_SPEED_THRESHOLD
        and abs(relative_vx) >= EVENT_LATERAL_SPEED_THRESHOLD
    ):
        add_event(
            "lateral_motion_onset",
            motion_confidence,
        )

    # --------------------------------------------------------------
    # PERSISTENT DEPTH-STATE CONFIRMATION
    #
    # The current state plus the immediately preceding N-1 states
    # must agree. The state immediately before that run must differ,
    # otherwise this is not an onset.
    # --------------------------------------------------------------

    depth_sequence = [
        state["depth_state"]
        for state in previous_states
    ] + [depth_state]

    if len(depth_sequence) >= DEPTH_CONFIRMATION_STATES + 1:

        confirmed_run = depth_sequence[
            -DEPTH_CONFIRMATION_STATES:
        ]

        preceding_depth = depth_sequence[
            -DEPTH_CONFIRMATION_STATES - 1
        ]

        if (
            all(
                value == "approaching"
                for value in confirmed_run
            )
            and preceding_depth != "approaching"
        ):
            add_event(
                "approach_onset",
                motion_confidence,
            )

        if (
            all(
                value == "receding"
                for value in confirmed_run
            )
            and preceding_depth != "receding"
        ):
            add_event(
                "recede_onset",
                motion_confidence,
            )

    # --------------------------------------------------------------
    # MOTION ONSET
    # --------------------------------------------------------------

    motion_sequence = [
        state["motion_state"]
        for state in previous_states
    ] + [motion_state]

    if len(motion_sequence) >= DEPTH_CONFIRMATION_STATES + 1:

        previous_motion_run = motion_sequence[
            -DEPTH_CONFIRMATION_STATES - 1:-1
        ]

        if (
            all(
                value == "stationary"
                for value in previous_motion_run
            )
            and motion_state != "stationary"
        ):
            add_event(
                "motion_onset",
                motion_confidence,
            )

    # --------------------------------------------------------------
    # DIRECTION CHANGE
    # --------------------------------------------------------------

    previous_vx = causal_history[-1][
        "relative_motion"
    ]["vx"]

    if (
        abs(previous_vx) >= EVENT_LATERAL_SPEED_THRESHOLD
        and abs(relative_vx) >= EVENT_LATERAL_SPEED_THRESHOLD
        and previous_vx * relative_vx < 0.0
    ):
        add_event(
            "direction_change",
            motion_confidence,
        )

    return events


# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================


def build_temporal_states(
    tracking_data: Dict[str, Any],
    identity_map: Dict[str, str],
) -> Dict[str, Any]:

    fps, _, _ = get_metadata(tracking_data)

    trajectories = extract_trajectories(tracking_data)
    entity_metadata = get_entity_metadata(tracking_data)

    if not trajectories:
        raise ValueError("No valid trajectories found.")

    start_time = min(
        points[0]["timestamp"]
        for points in trajectories.values()
    )

    end_time = max(
        points[-1]["timestamp"]
        for points in trajectories.values()
    )

    entity_states = {
        entity_id: []
        for entity_id in trajectories
    }

    scene_timeline = []
    scene_flow_history = []

    current_time = start_time

    while current_time <= end_time + 1e-9:

        active_entities = {}

        for entity_id, trajectory in trajectories.items():

            window = get_causal_window(
                trajectory,
                current_time,
            )

            if not window:
                continue

            latest = window[-1]

            if (
                current_time - latest["timestamp"]
                > MAX_POINT_STALENESS_SECONDS
            ):
                continue

            raw_motion = calculate_raw_motion(window)

            active_entities[entity_id] = {
                "window": window,
                "latest": latest,
                "raw_motion": raw_motion,
            }

        raw_scene_flow = estimate_scene_flow(
            active_entities
        )

        scene_flow = smooth_scene_flow(
            scene_flow_history,
            raw_scene_flow,
            current_time,
        )

        history_item = {
            key: scene_flow[key]
            for key in SCENE_FLOW_KEYS
        }

        history_item.update(
            {
                "time_seconds": current_time,
                "reliable": scene_flow["reliable"],
                "confidence": scene_flow["confidence"],
            }
        )

        scene_flow_history.append(history_item)

        scene_flow_history = [
            item
            for item in scene_flow_history
            if (
                current_time - item["time_seconds"]
                <= SCENE_FLOW_SMOOTHING_SECONDS
            )
        ]

        scene_timeline.append(
            {
                "time_seconds": round(current_time, 3),
                "reliable": scene_flow["reliable"],
                "confidence": round(
                    scene_flow["confidence"],
                    4,
                ),
                "residual_agreement": round(
                    scene_flow.get(
                        "residual_agreement",
                        0.0,
                    ),
                    4,
                ),
                "temporal_consistency": round(
                    scene_flow.get(
                        "temporal_consistency",
                        0.0,
                    ),
                    4,
                ),
                "discontinuity_blocked": scene_flow.get(
                    "discontinuity_blocked",
                    False,
                ),
                "entities_used": scene_flow["sample_count"],
                **{
                    key: round(scene_flow[key], 6)
                    for key in SCENE_FLOW_KEYS
                },
            }
        )

        for entity_id, active in active_entities.items():

            raw_motion = active["raw_motion"]

            if not raw_motion["reliable"]:
                continue

            latest = active["latest"]

            scene_motion = expected_scene_motion(
                latest,
                scene_flow,
            )

            relative_vx = (
                raw_motion["vx"]
                - scene_motion["vx"]
            )

            relative_vy = (
                raw_motion["vy"]
                - scene_motion["vy"]
            )

            relative_area_rate = (
                raw_motion["area_rate"]
                - scene_motion["area_rate"]
            )

            motion_quality = calculate_motion_quality(
                active["window"],
                relative_vx,
                relative_vy,
            )

            acceleration = calculate_acceleration(
                active["window"]
            )

            depth_state, depth_confidence = classify_depth_state(
                relative_vy,
                relative_area_rate,
            )

            depth_confidence *= (
                0.55
                + 0.45
                * motion_quality["motion_confidence"]
            )

            depth_confidence = clamp(
                depth_confidence
            )

            motion_state = classify_motion_state(
                relative_vx,
                relative_vy,
                relative_area_rate,
            )

            ego_path = calculate_ego_path_evidence(
                latest["bbox"],
                relative_vx,
                motion_quality["motion_confidence"],
            )

            previous_states = entity_states[entity_id]

            motion_events = detect_motion_events(
                previous_states,
                motion_state,
                depth_state,
                relative_vx,
                relative_vy,
                relative_area_rate,
                motion_quality["motion_confidence"],
                current_time,
            )

            metadata = entity_metadata.get(
                entity_id,
                {},
            )

            state = {
                "time_seconds": round(current_time, 3),
                "frame": int(round(current_time * fps)),
                "entity_id": entity_id,
                "visual_identity": identity_map.get(
                    entity_id,
                    entity_id,
                ),
                "class_name": get_class_name(metadata),
                "position": classify_position(
                    latest["center_x"],
                    latest["center_y"],
                ),
                "geometry": {
                    "center_x": round(latest["center_x"], 6),
                    "center_y": round(latest["center_y"], 6),
                    "bbox": [
                        round(value, 6)
                        for value in latest["bbox"]
                    ],
                    "width": round(latest["width"], 6),
                    "height": round(latest["height"], 6),
                    "area": round(latest["area"], 6),
                },
                "raw_motion": {
                    "vx": round(raw_motion["vx"], 6),
                    "vy": round(raw_motion["vy"], 6),
                    "width_rate": round(
                        raw_motion["width_rate"],
                        6,
                    ),
                    "height_rate": round(
                        raw_motion["height_rate"],
                        6,
                    ),
                    "area_rate": round(
                        raw_motion["area_rate"],
                        6,
                    ),
                },
                "scene_motion": {
                    "vx": round(scene_motion["vx"], 6),
                    "vy": round(scene_motion["vy"], 6),
                    "area_rate": round(
                        scene_motion["area_rate"],
                        6,
                    ),
                    "reliable": scene_flow["reliable"],
                    "confidence": round(
                        scene_flow["confidence"],
                        4,
                    ),
                },
                "relative_motion": {
                    "vx": round(relative_vx, 6),
                    "vy": round(relative_vy, 6),
                    "area_rate": round(
                        relative_area_rate,
                        6,
                    ),
                    "speed": round(
                        math.hypot(
                            relative_vx,
                            relative_vy,
                        ),
                        6,
                    ),
                },
                "acceleration": {
                    "ax": round(acceleration["ax"], 6),
                    "ay": round(acceleration["ay"], 6),
                    "magnitude": round(
                        acceleration["magnitude"],
                        6,
                    ),
                    "normalized_magnitude": round(
                        clamp(
                            acceleration["magnitude"]
                            / ACCELERATION_SCALE
                        ),
                        4,
                    ),
                },
                "motion_state": motion_state,
                "depth_state": depth_state,
                "depth_confidence": round(
                    depth_confidence,
                    4,
                ),
                "motion_quality": {
                    "duration": round(
                        motion_quality["duration"],
                        4,
                    ),
                    "velocity_stability": round(
                        motion_quality[
                            "velocity_stability"
                        ],
                        4,
                    ),
                    "direction_persistence": round(
                        motion_quality[
                            "direction_persistence"
                        ],
                        4,
                    ),
                    "motion_confidence": round(
                        motion_quality[
                            "motion_confidence"
                        ],
                        4,
                    ),
                    "reliability": motion_quality[
                        "reliability"
                    ],
                },
                "ego_path": {
                    "currently_inside": ego_path[
                        "currently_inside"
                    ],
                    "overlap_ratio": round(
                        ego_path["overlap_ratio"],
                        4,
                    ),
                    "lateral_distance_to_corridor": round(
                        ego_path[
                            "lateral_distance_to_corridor"
                        ],
                        6,
                    ),
                    "moving_toward_corridor": ego_path[
                        "moving_toward_corridor"
                    ],
                    "intrusion_rate": round(
                        ego_path["intrusion_rate"],
                        6,
                    ),
                    "confidence": round(
                        ego_path["confidence"],
                        4,
                    ),
                },
                "motion_events": motion_events,
            }

            entity_states[entity_id].append(state)

        current_time += TIMELINE_STEP_SECONDS

    return {
        "configuration": {
            "causal": True,
            "timeline_step_seconds": TIMELINE_STEP_SECONDS,
            "motion_window_seconds": MOTION_WINDOW_SECONDS,
            "scene_compensation": True,
            "scene_flow_temporal_smoothing": True,
            "scene_flow_smoothing_seconds": (
                SCENE_FLOW_SMOOTHING_SECONDS
            ),
            "scene_flow_confidence": True,
            "motion_reliability_calibration": True,
            "ego_path_evidence": True,
            "motion_transition_events": True,
            "motion_event_lookback_seconds": EVENT_LOOKBACK_SECONDS,
            "motion_event_history_based": True,
            "depth_confirmation_states": DEPTH_CONFIRMATION_STATES,
            "event_cooldown_seconds": EVENT_COOLDOWN_SECONDS,
            "entity_relative_speed_transitions": True,
            "scene_flow_temporal_authority_gate": True,
            "scene_flow_min_temporal_consistency": 0.15,
            "accident_classification": False,
            "bbox_overlap_is_collision": False,
            "design_note": (
                "This stage produces causal single-entity motion, "
                "depth, reliability, ego-corridor evidence, and "
                "motion-transition events. Bounding-box overlap is "
                "not collision evidence. Pairwise depth-compatible "
                "trajectory conflict must be evaluated downstream."
            ),
        },
        "scene_flow_timeline": scene_timeline,
        "entity_states": entity_states,
    }


# ============================================================================
# SUMMARY
# ============================================================================


def build_entity_summaries(
    result: Dict[str, Any],
) -> List[Dict[str, Any]]:

    summaries = []

    for (
        entity_id,
        states,
    ) in result[
        "entity_states"
    ].items():

        if not states:
            continue

        motion_states = [
            state["motion_state"]
            for state in states
        ]

        depth_states = [
            state["depth_state"]
            for state in states
        ]

        dominant_motion = max(
            set(motion_states),
            key=motion_states.count,
        )

        dominant_depth = max(
            set(depth_states),
            key=depth_states.count,
        )

        average_confidence = float(
            np.mean(
                [
                    state[
                        "motion_quality"
                    ][
                        "motion_confidence"
                    ]
                    for state in states
                ]
            )
        )

        average_stability = float(
            np.mean(
                [
                    state[
                        "motion_quality"
                    ][
                        "velocity_stability"
                    ]
                    for state in states
                ]
            )
        )

        maximum_speed = max(
            state[
                "relative_motion"
            ]["speed"]
            for state in states
        )

        maximum_acceleration = max(
            state[
                "acceleration"
            ]["magnitude"]
            for state in states
        )

        reliability_levels = [
            state["motion_quality"]["reliability"]
            for state in states
        ]

        reliability_rank = {
            "low": 0,
            "moderate": 1,
            "high": 2,
        }

        dominant_reliability = max(
            set(reliability_levels),
            key=reliability_levels.count,
        )

        motion_events = [
            event
            for state in states
            for event in state.get(
                "motion_events",
                [],
            )
        ]

        ego_intrusion_states = sum(
            1
            for state in states
            if state.get(
                "ego_path",
                {},
            ).get(
                "moving_toward_corridor",
                False,
            )
        )

        summaries.append(
            {
                "entity_id": entity_id,
                "visual_identity": (
                    states[0][
                        "visual_identity"
                    ]
                ),
                "class_name": (
                    states[0][
                        "class_name"
                    ]
                ),
                "states_generated": len(
                    states
                ),
                "first_state_seconds": (
                    states[0][
                        "time_seconds"
                    ]
                ),
                "last_state_seconds": (
                    states[-1][
                        "time_seconds"
                    ]
                ),
                "dominant_motion_state": (
                    dominant_motion
                ),
                "dominant_depth_state": (
                    dominant_depth
                ),
                "average_motion_confidence": round(
                    average_confidence,
                    4,
                ),
                "average_velocity_stability": round(
                    average_stability,
                    4,
                ),
                "maximum_relative_speed": round(
                    maximum_speed,
                    6,
                ),
                "maximum_acceleration": round(
                    maximum_acceleration,
                    6,
                ),
                "dominant_motion_reliability": (
                    dominant_reliability
                ),
                "motion_events": motion_events,
                "motion_event_count": len(
                    motion_events
                ),
                "ego_intrusion_states": (
                    ego_intrusion_states
                ),
            }
        )

    return summaries


# ============================================================================
# OUTPUT
# ============================================================================


def derive_output_path(
    tracking_path: Path,
) -> Path:

    stem = tracking_path.stem

    if stem.endswith(
        "_tracking"
    ):

        stem = stem[
            :-len("_tracking")
        ]

    return tracking_path.with_name(
        f"{stem}_object_analysis.json"
    )


def print_summary(
    result: Dict[str, Any],
) -> None:

    summaries = result[
        "entity_summaries"
    ]

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TEMPORAL OBJECT ANALYSIS COMPLETE"
    )

    print(
        "=" * 72
    )

    for summary in summaries:

        print(
            f"\n{summary['entity_id']} "
            f"({summary['visual_identity']})"
        )

        print(
            f"  Class       : "
            f"{summary['class_name']}"
        )

        print(
            f"  States      : "
            f"{summary['states_generated']}"
        )

        print(
            f"  Motion      : "
            f"{summary['dominant_motion_state']}"
        )

        print(
            f"  Depth       : "
            f"{summary['dominant_depth_state']}"
        )

        print(
            f"  Confidence  : "
            f"{summary['average_motion_confidence']:.3f}"
        )

        print(
            f"  Stability   : "
            f"{summary['average_velocity_stability']:.3f}"
        )

        print(
            f"  Max speed   : "
            f"{summary['maximum_relative_speed']:.6f}"
        )

        print(
            f"  Max accel   : "
            f"{summary['maximum_acceleration']:.6f}"
        )

    print(
        "\n"
        + "-" * 72
    )

    print(
        f"Entities analyzed       : "
        f"{len(summaries)}"
    )

    print(
        f"Temporal states created : "
        f"{sum(item['states_generated'] for item in summaries)}"
    )

    print(
        "Accident classification : DISABLED"
    )

    print(
        "BBox overlap = collision : FALSE"
    )

    print(
        "=" * 72
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract causal temporal motion "
            "states for tracked road objects."
        )
    )

    parser.add_argument(
        "tracking_json",
        type=Path,
    )

    parser.add_argument(
        "--identity_json",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    print(
        "[analyzer] Loading tracking data..."
    )

    with args.tracking_json.open(
        "r",
        encoding="utf-8",
    ) as file:

        tracking_data = json.load(
            file
        )

    identity_map = load_identity_map(
        args.identity_json
    )

    trajectories = extract_trajectories(
        tracking_data
    )

    print(
        f"[analyzer] Preparing "
        f"{len(trajectories)} physical entities..."
    )

    print(
        "[analyzer] Building causal temporal states..."
    )

    result = build_temporal_states(
        tracking_data,
        identity_map,
    )

    result[
        "entity_summaries"
    ] = build_entity_summaries(
        result
    )

    output_path = (
        args.output
        if args.output is not None
        else derive_output_path(
            args.tracking_json
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print_summary(
        result
    )

    print(
        f"\nJSON written to: "
        f"{output_path}"
    )


if __name__ == "__main__":

    main()
