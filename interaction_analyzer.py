"""
interaction_analyzer_v6.py

Standalone Stage-3 V6 pairwise conflict-risk analyzer.

V6 replaces "best trigger + nearby motion response" with:
    synchronized pair state
        -> interaction topology
        -> pairwise risk timeline R(t)
        -> joint transition search
        -> pair-conditioned response validation
        -> risk reduction / trajectory discontinuity
        -> critical-event candidate

Important:
- Image-space quantities are treated as proxies, not metric speed/distance.
- Bounding-box overlap is not collision proof.
- A generic motion event is not inherently evasive.
- A response only supports a pair when it changes that pair's risk geometry.
- This stage does not confirm an accident, injury, or fault.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# ROAD USERS
# ============================================================================

SUPPORTED_ROAD_USERS = {
    "car", "truck", "bus",
    "motorcycle", "motorbike",
    "bicycle", "bike",
    "person", "pedestrian",
}

VRU_CLASSES = {
    "motorcycle", "motorbike",
    "bicycle", "bike",
    "person", "pedestrian",
}

PEDESTRIAN_CLASSES = {"person", "pedestrian"}
TWO_WHEELER_CLASSES = {"motorcycle", "motorbike", "bicycle", "bike"}


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_PAIR_CENTER_DISTANCE = 0.55
MAX_PAIR_EDGE_DISTANCE = 0.24

DEPTH_CENTER_Y_TOLERANCE = 0.20
MAX_LOG_AREA_RATIO_VEHICLE = 1.65
MAX_LOG_AREA_RATIO_VRU = 2.90
MIN_DEPTH_SCORE = 0.16

PREDICTION_HORIZON_SECONDS = 1.20
MIN_RELATIVE_SPEED = 0.010

EPISODE_MAX_GAP_SECONDS = 0.21
EPISODE_MIN_STATES = 3
EPISODE_MIN_PEAK_RISK = 0.20

RISK_LOOKBACK_SECONDS = 0.60
RESPONSE_HORIZON_SECONDS = 0.90
MIN_RESPONSE_DELAY_SECONDS = 0.0

MIN_PEAK_RISK = 0.43
MIN_RISK_ESCALATION = 0.12
MIN_PAIR_RESPONSE = 0.18
MIN_RISK_TRANSITION = 0.12
MIN_EVENT_SCORE = 0.50

MIN_PRE_STATES = 2
MIN_POST_STATES = 1

RISK_SMOOTHING_WINDOW = 3

# Risk scales are image-space proxy scales, not physical units.
NEAR_EDGE_SCALE = 0.10
CENTER_DISTANCE_SCALE = 0.22
CLOSING_SPEED_SCALE = 0.10
MISS_DISTANCE_SCALE = 0.11
TTC_URGENT_SECONDS = 1.20
TTC_FLOOR_SECONDS = 0.15

# Pair-conditioned response scales.
SPEED_CHANGE_SCALE = 0.08
DIRECTION_CHANGE_RADIANS = math.radians(55.0)
LATERAL_CHANGE_SCALE = 0.07
ACCELERATION_SCALE = 0.25

# Topology thresholds.
PARALLEL_COSINE = 0.72
OPPOSING_COSINE = -0.55
CROSSING_COSINE_ABS = 0.55
LATERAL_DOMINANCE = 1.15

# Prevent one entity's generic response from creating many unrelated events.
MAX_CRITICAL_PAIRS_PER_ENTITY_WINDOW = 2
DUPLICATE_EVENT_TIME_WINDOW = 0.65


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_or_zero(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median_or_zero(values: List[float]) -> float:
    return float(np.median(values)) if values else 0.0


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def pair_key(entity_a: str, entity_b: str) -> Tuple[str, str]:
    return tuple(sorted((str(entity_a), str(entity_b))))


def vector_norm(x: float, y: float) -> float:
    return math.hypot(x, y)


def cosine_similarity(ax: float, ay: float, bx: float, by: float) -> float:
    na = vector_norm(ax, ay)
    nb = vector_norm(bx, by)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return clamp((ax * bx + ay * by) / (na * nb), -1.0, 1.0)


def angle_change(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    n1 = vector_norm(*v1)
    n2 = vector_norm(*v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cosine = clamp(
        (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2),
        -1.0,
        1.0,
    )
    return math.acos(cosine)


def bbox_edge_distance(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return math.hypot(dx, dy)


def bbox_overlap_ratio(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-8)
    area_b = max((bx2 - bx1) * (by2 - by1), 1e-8)
    return intersection / min(area_a, area_b)


def is_supported_entity(state: Dict[str, Any]) -> bool:
    return str(state.get("class_name", "")).lower() in SUPPORTED_ROAD_USERS


def is_vru_class(class_name: str) -> bool:
    return str(class_name).lower() in VRU_CLASSES


def pair_has_vru(state_a: Dict[str, Any], state_b: Dict[str, Any]) -> bool:
    return is_vru_class(state_a.get("class_name", "")) or is_vru_class(
        state_b.get("class_name", "")
    )


# ============================================================================
# INPUT INDEXING
# ============================================================================

def build_time_index(data: Dict[str, Any]) -> Dict[float, List[Dict[str, Any]]]:
    result: Dict[float, List[Dict[str, Any]]] = defaultdict(list)

    for states in data.get("entity_states", {}).values():
        if not isinstance(states, list):
            continue
        for state in states:
            if not isinstance(state, dict) or not is_supported_entity(state):
                continue
            time_seconds = round(safe_float(state.get("time_seconds")), 3)
            result[time_seconds].append(state)

    return dict(sorted(result.items()))


def build_entity_history(
    data: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    result = {}
    for entity_id, states in data.get("entity_states", {}).items():
        if not isinstance(states, list):
            continue
        result[str(entity_id)] = sorted(
            [state for state in states if isinstance(state, dict)],
            key=lambda state: safe_float(state.get("time_seconds")),
        )
    return result


# ============================================================================
# STATE ACCESS
# ============================================================================

def entity_id(state: Dict[str, Any]) -> str:
    return str(state.get("entity_id", state.get("visual_identity", "unknown")))


def geometry(state: Dict[str, Any]) -> Dict[str, Any]:
    return state.get("geometry", {})


def motion(state: Dict[str, Any]) -> Dict[str, Any]:
    return state.get("relative_motion", {})


def acceleration(state: Dict[str, Any]) -> Dict[str, Any]:
    return state.get("acceleration", {})


def center(state: Dict[str, Any]) -> Tuple[float, float]:
    g = geometry(state)
    return safe_float(g.get("center_x")), safe_float(g.get("center_y"))


def bbox(state: Dict[str, Any]) -> List[float]:
    value = geometry(state).get("bbox", [0.0, 0.0, 0.0, 0.0])
    if not isinstance(value, list) or len(value) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [safe_float(item) for item in value]


def velocity(state: Dict[str, Any]) -> Tuple[float, float]:
    m = motion(state)
    return safe_float(m.get("vx")), safe_float(m.get("vy"))


def speed(state: Dict[str, Any]) -> float:
    m = motion(state)
    if "speed" in m:
        return safe_float(m.get("speed"))
    return vector_norm(*velocity(state))


def acceleration_magnitude(state: Dict[str, Any]) -> float:
    return safe_float(acceleration(state).get("magnitude"))


def motion_reliability(state: Dict[str, Any]) -> float:
    quality = state.get("motion_quality", {})
    confidence = clamp(safe_float(quality.get("motion_confidence")))
    stability = clamp(safe_float(quality.get("velocity_stability")))
    reliability_name = str(quality.get("reliability", "")).lower()

    reliability_multiplier = {
        "low": 0.45,
        "moderate": 0.75,
        "high": 1.0,
    }.get(reliability_name, 0.80)

    # velocity_stability in the current object analyzer behaves more like
    # a consistency score in summaries, so use it as positive evidence.
    return clamp(
        confidence
        * reliability_multiplier
        * (0.70 + 0.30 * stability)
    )


# ============================================================================
# DEPTH COMPATIBILITY
# ============================================================================

def calculate_depth_compatibility(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
) -> Dict[str, Any]:
    ax, ay = center(state_a)
    bx, by = center(state_b)

    area_a = max(safe_float(geometry(state_a).get("area")), 1e-8)
    area_b = max(safe_float(geometry(state_b).get("area")), 1e-8)

    center_y_gap = abs(ay - by)
    log_area_ratio = abs(math.log(area_a / area_b))
    vulnerable_pair = pair_has_vru(state_a, state_b)

    max_scale_gap = (
        MAX_LOG_AREA_RATIO_VRU
        if vulnerable_pair
        else MAX_LOG_AREA_RATIO_VEHICLE
    )

    vertical_score = clamp(1.0 - center_y_gap / DEPTH_CENTER_Y_TOLERANCE)
    scale_score = clamp(1.0 - log_area_ratio / max(max_scale_gap, 1e-6))

    depth_state_a = str(state_a.get("depth_state", "stable"))
    depth_state_b = str(state_b.get("depth_state", "stable"))
    depth_trend_score = (
        1.0
        if depth_state_a == depth_state_b
        else 0.55
        if "stable" in {depth_state_a, depth_state_b}
        else 0.25
    )

    score = clamp(
        0.46 * vertical_score
        + 0.39 * scale_score
        + 0.15 * depth_trend_score
    )

    return {
        "compatible": score >= MIN_DEPTH_SCORE,
        "score": score,
        "center_y_gap": center_y_gap,
        "log_area_ratio": log_area_ratio,
        "vulnerable_road_user_pair": vulnerable_pair,
    }


# ============================================================================
# PAIR KINEMATICS
# ============================================================================

def calculate_pair_kinematics(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
) -> Dict[str, Any]:
    ax, ay = center(state_a)
    bx, by = center(state_b)
    avx, avy = velocity(state_a)
    bvx, bvy = velocity(state_b)

    rx = bx - ax
    ry = by - ay
    rvx = bvx - avx
    rvy = bvy - avy

    center_distance = vector_norm(rx, ry)
    relative_speed = vector_norm(rvx, rvy)

    if center_distance > 1e-8:
        closing_speed = max(
            0.0,
            -((rx * rvx + ry * rvy) / center_distance),
        )
    else:
        closing_speed = relative_speed

    time_to_closest = None
    predicted_min_distance = center_distance

    relative_speed_sq = rvx * rvx + rvy * rvy
    if relative_speed_sq >= MIN_RELATIVE_SPEED ** 2:
        t_star = -((rx * rvx + ry * rvy) / relative_speed_sq)
        if 0.0 <= t_star <= PREDICTION_HORIZON_SECONDS:
            time_to_closest = t_star
            predicted_min_distance = vector_norm(
                rx + rvx * t_star,
                ry + rvy * t_star,
            )

    return {
        "center_distance": center_distance,
        "edge_distance": bbox_edge_distance(bbox(state_a), bbox(state_b)),
        "bbox_overlap_ratio": bbox_overlap_ratio(bbox(state_a), bbox(state_b)),
        "relative_speed": relative_speed,
        "closing_speed": closing_speed,
        "time_to_closest_proxy": time_to_closest,
        "predicted_min_center_distance": predicted_min_distance,
        "relative_position": [rx, ry],
        "relative_velocity": [rvx, rvy],
    }


# ============================================================================
# INTERACTION TOPOLOGY
# ============================================================================

def classify_interaction_topology(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
    kin: Dict[str, Any],
) -> Dict[str, Any]:
    class_a = str(state_a.get("class_name", "")).lower()
    class_b = str(state_b.get("class_name", "")).lower()

    avx, avy = velocity(state_a)
    bvx, bvy = velocity(state_b)

    speed_a = vector_norm(avx, avy)
    speed_b = vector_norm(bvx, bvy)
    cosine = cosine_similarity(avx, avy, bvx, bvy)

    rx, ry = kin["relative_position"]
    horizontal_separation = abs(rx)
    vertical_separation = abs(ry)

    has_pedestrian = (
        class_a in PEDESTRIAN_CLASSES
        or class_b in PEDESTRIAN_CLASSES
    )
    has_two_wheeler = (
        class_a in TWO_WHEELER_CLASSES
        or class_b in TWO_WHEELER_CLASSES
    )

    lateral_a = abs(avx) > LATERAL_DOMINANCE * abs(avy)
    lateral_b = abs(bvx) > LATERAL_DOMINANCE * abs(bvy)

    if has_pedestrian:
        topology = "VRU_CROSSING"
        confidence = clamp(
            0.45
            + 0.30 * max(float(lateral_a), float(lateral_b))
            + 0.25 * clamp(kin["closing_speed"] / CLOSING_SPEED_SCALE)
        )

    elif has_two_wheeler and (
        lateral_a
        or lateral_b
        or horizontal_separation < 0.16
    ):
        topology = "LATERAL_INTRUSION"
        confidence = clamp(
            0.45
            + 0.25 * max(float(lateral_a), float(lateral_b))
            + 0.30 * clamp(
                1.0 - kin["edge_distance"] / MAX_PAIR_EDGE_DISTANCE
            )
        )

    elif speed_a < MIN_RELATIVE_SPEED or speed_b < MIN_RELATIVE_SPEED:
        topology = "LATERAL_INTRUSION"
        confidence = clamp(
            0.35
            + 0.35 * max(float(lateral_a), float(lateral_b))
            + 0.30 * clamp(kin["closing_speed"] / CLOSING_SPEED_SCALE)
        )

    elif cosine <= OPPOSING_COSINE:
        topology = "HEAD_ON"
        confidence = clamp(abs(cosine))

    elif abs(cosine) <= CROSSING_COSINE_ABS:
        topology = "CROSSING"
        confidence = clamp(
            0.55
            + 0.45 * (1.0 - abs(cosine) / CROSSING_COSINE_ABS)
        )

    elif cosine >= PARALLEL_COSINE:
        if horizontal_separation > vertical_separation * 0.80:
            topology = "MERGING"
        else:
            topology = "FOLLOWING"
        confidence = clamp(cosine)

    else:
        topology = "UNKNOWN"
        confidence = 0.35

    return {
        "type": topology,
        "confidence": confidence,
        "velocity_cosine": cosine,
        "horizontal_separation": horizontal_separation,
        "vertical_separation": vertical_separation,
    }


# ============================================================================
# PAIRWISE RISK
# ============================================================================

def calculate_risk_components(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
    kin: Dict[str, Any],
    depth: Dict[str, Any],
    topology: Dict[str, Any],
) -> Dict[str, float]:
    edge_risk = clamp(1.0 - kin["edge_distance"] / NEAR_EDGE_SCALE)
    center_risk = clamp(1.0 - kin["center_distance"] / CENTER_DISTANCE_SCALE)
    closing_risk = clamp(kin["closing_speed"] / CLOSING_SPEED_SCALE)
    miss_risk = clamp(
        1.0
        - kin["predicted_min_center_distance"] / MISS_DISTANCE_SCALE
    )

    ttc = kin["time_to_closest_proxy"]
    if ttc is None:
        ttc_risk = 0.0
    else:
        ttc_risk = clamp(
            (TTC_URGENT_SECONDS - max(ttc, TTC_FLOOR_SECONDS))
            / (TTC_URGENT_SECONDS - TTC_FLOOR_SECONDS)
        )

    ego_a = state_a.get("ego_path", {})
    ego_b = state_b.get("ego_path", {})
    intrusion_risk = max(
        float(bool(ego_a.get("moving_toward_corridor", False))),
        float(bool(ego_b.get("moving_toward_corridor", False))),
        0.55 * float(bool(ego_a.get("currently_inside", False))),
        0.55 * float(bool(ego_b.get("currently_inside", False))),
    )

    depth_score = clamp(depth["score"])
    topology_confidence = clamp(topology["confidence"])

    topology_type = topology["type"]

    if topology_type == "VRU_CROSSING":
        raw_risk = (
            0.18 * edge_risk
            + 0.10 * center_risk
            + 0.19 * closing_risk
            + 0.22 * miss_risk
            + 0.16 * ttc_risk
            + 0.15 * intrusion_risk
        )
        topology_prior = 0.10

    elif topology_type == "LATERAL_INTRUSION":
        raw_risk = (
            0.20 * edge_risk
            + 0.10 * center_risk
            + 0.14 * closing_risk
            + 0.20 * miss_risk
            + 0.12 * ttc_risk
            + 0.24 * intrusion_risk
        )
        topology_prior = 0.06

    elif topology_type in {"CROSSING", "HEAD_ON"}:
        raw_risk = (
            0.14 * edge_risk
            + 0.08 * center_risk
            + 0.22 * closing_risk
            + 0.25 * miss_risk
            + 0.23 * ttc_risk
            + 0.08 * intrusion_risk
        )
        topology_prior = 0.05

    elif topology_type in {"FOLLOWING", "MERGING"}:
        raw_risk = (
            0.21 * edge_risk
            + 0.10 * center_risk
            + 0.22 * closing_risk
            + 0.19 * miss_risk
            + 0.20 * ttc_risk
            + 0.08 * intrusion_risk
        )
        topology_prior = 0.02

    else:
        raw_risk = (
            0.18 * edge_risk
            + 0.12 * center_risk
            + 0.22 * closing_risk
            + 0.22 * miss_risk
            + 0.18 * ttc_risk
            + 0.08 * intrusion_risk
        )
        topology_prior = 0.0

    reliability = math.sqrt(
        max(motion_reliability(state_a), 0.05)
        * max(motion_reliability(state_b), 0.05)
    )

    # Depth does not hard-gate a VRU pair; monocular scale mismatch is expected.
    depth_factor = (
        0.58 + 0.42 * depth_score
        if pair_has_vru(state_a, state_b)
        else 0.35 + 0.65 * depth_score
    )

    reliability_factor = 0.55 + 0.45 * reliability

    risk = clamp(
        (raw_risk + topology_prior * topology_confidence)
        * depth_factor
        * reliability_factor
    )

    return {
        "edge_proximity": edge_risk,
        "center_proximity": center_risk,
        "closing": closing_risk,
        "predicted_miss": miss_risk,
        "time_to_closest": ttc_risk,
        "ego_intrusion": intrusion_risk,
        "depth_support": depth_score,
        "motion_reliability": reliability,
        "topology_support": topology_confidence,
        "risk": risk,
    }


# ============================================================================
# PAIR STATE
# ============================================================================

def analyze_pair_state(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
    current_time: float,
) -> Optional[Dict[str, Any]]:
    ax, ay = center(state_a)
    bx, by = center(state_b)
    center_distance = vector_norm(bx - ax, by - ay)

    edge_distance = bbox_edge_distance(bbox(state_a), bbox(state_b))

    if (
        center_distance > MAX_PAIR_CENTER_DISTANCE
        and edge_distance > MAX_PAIR_EDGE_DISTANCE
    ):
        return None

    kin = calculate_pair_kinematics(state_a, state_b)
    depth = calculate_depth_compatibility(state_a, state_b)
    topology = classify_interaction_topology(state_a, state_b, kin)
    risk_components = calculate_risk_components(
        state_a,
        state_b,
        kin,
        depth,
        topology,
    )

    entity_a = entity_id(state_a)
    entity_b = entity_id(state_b)

    return {
        "time_seconds": round(current_time, 3),
        "entity_a": entity_a,
        "entity_b": entity_b,
        "visual_identity_a": state_a.get("visual_identity", entity_a),
        "visual_identity_b": state_b.get("visual_identity", entity_b),
        "class_a": state_a.get("class_name", "object"),
        "class_b": state_b.get("class_name", "object"),
        "interaction_topology": {
            "type": topology["type"],
            "confidence": round(topology["confidence"], 4),
            "velocity_cosine": round(topology["velocity_cosine"], 4),
        },
        "risk_score": round(risk_components["risk"], 4),
        "risk_components": {
            key: round(value, 4)
            for key, value in risk_components.items()
            if key != "risk"
        },
        "geometry": {
            "center_distance": round(kin["center_distance"], 6),
            "edge_distance": round(kin["edge_distance"], 6),
            "bbox_overlap_ratio": round(kin["bbox_overlap_ratio"], 4),
            "note": (
                "Image-space geometry is risk context only. "
                "Bounding-box overlap is not collision confirmation."
            ),
        },
        "trajectory_proxy": {
            "relative_speed": round(kin["relative_speed"], 6),
            "closing_speed": round(kin["closing_speed"], 6),
            "time_to_closest_proxy": (
                round(kin["time_to_closest_proxy"], 4)
                if kin["time_to_closest_proxy"] is not None
                else None
            ),
            "predicted_min_center_distance": round(
                kin["predicted_min_center_distance"], 6
            ),
            "note": "Image-space trajectory proxy; not metric TTC or distance.",
        },
        "depth_compatibility": {
            "compatible": depth["compatible"],
            "score": round(depth["score"], 4),
            "center_y_gap": round(depth["center_y_gap"], 6),
            "log_area_ratio": round(depth["log_area_ratio"], 6),
            "vulnerable_road_user_pair": depth[
                "vulnerable_road_user_pair"
            ],
        },
        "participant_motion": {
            entity_a: {
                "vx": round(velocity(state_a)[0], 6),
                "vy": round(velocity(state_a)[1], 6),
                "speed": round(speed(state_a), 6),
                "acceleration": round(acceleration_magnitude(state_a), 6),
                "motion_confidence": round(motion_reliability(state_a), 4),
            },
            entity_b: {
                "vx": round(velocity(state_b)[0], 6),
                "vy": round(velocity(state_b)[1], 6),
                "speed": round(speed(state_b), 6),
                "acceleration": round(acceleration_magnitude(state_b), 6),
                "motion_confidence": round(motion_reliability(state_b), 4),
            },
        },
    }


# ============================================================================
# RISK TIMELINE
# ============================================================================

def smooth_pair_risk(states: List[Dict[str, Any]]) -> None:
    raw = [safe_float(state.get("risk_score")) for state in states]

    for index, state in enumerate(states):
        start = max(0, index - RISK_SMOOTHING_WINDOW + 1)
        smoothed = float(np.median(raw[start:index + 1]))
        state["smoothed_risk_score"] = round(smoothed, 4)

        if index == 0:
            state["risk_delta"] = 0.0
        else:
            state["risk_delta"] = round(
                smoothed
                - safe_float(states[index - 1].get("smoothed_risk_score")),
                4,
            )


def build_interaction_timeline(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    time_index = build_time_index(data)
    timeline = []
    pair_states: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for current_time, states in time_index.items():
        interactions = []

        for index_a in range(len(states)):
            for index_b in range(index_a + 1, len(states)):
                interaction = analyze_pair_state(
                    states[index_a],
                    states[index_b],
                    current_time,
                )
                if interaction is None:
                    continue
                interactions.append(interaction)
                pair_states[
                    pair_key(
                        interaction["entity_a"],
                        interaction["entity_b"],
                    )
                ].append(interaction)

        interactions.sort(
            key=lambda item: item["risk_score"],
            reverse=True,
        )

        timeline.append(
            {
                "time_seconds": current_time,
                "active_interactions": interactions,
            }
        )

    for states in pair_states.values():
        states.sort(key=lambda item: item["time_seconds"])
        smooth_pair_risk(states)

    # pair_states contains the same dict objects stored in timeline.
    for timeline_state in timeline:
        timeline_state["active_interactions"].sort(
            key=lambda item: item.get("smoothed_risk_score", item["risk_score"]),
            reverse=True,
        )

    return timeline


# ============================================================================
# EPISODE GROUPING
# ============================================================================

def collect_pair_states(
    timeline: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    result: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for timeline_state in timeline:
        for interaction in timeline_state["active_interactions"]:
            result[
                pair_key(
                    interaction["entity_a"],
                    interaction["entity_b"],
                )
            ].append(interaction)

    for states in result.values():
        states.sort(key=lambda state: state["time_seconds"])

    return result


def split_into_episodes(
    states: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    if not states:
        return []

    episodes = []
    current = [states[0]]

    for state in states[1:]:
        gap = state["time_seconds"] - current[-1]["time_seconds"]

        if gap <= EPISODE_MAX_GAP_SECONDS:
            current.append(state)
        else:
            if (
                len(current) >= EPISODE_MIN_STATES
                and max(
                    safe_float(item.get("smoothed_risk_score"))
                    for item in current
                ) >= EPISODE_MIN_PEAK_RISK
            ):
                episodes.append(current)
            current = [state]

    if (
        len(current) >= EPISODE_MIN_STATES
        and max(
            safe_float(item.get("smoothed_risk_score"))
            for item in current
        ) >= EPISODE_MIN_PEAK_RISK
    ):
        episodes.append(current)

    return episodes


# ============================================================================
# PAIR-CONDITIONED RESPONSE
# ============================================================================

def participant_motion_at(
    pair_state: Dict[str, Any],
    participant_id: str,
) -> Dict[str, float]:
    return pair_state.get("participant_motion", {}).get(
        participant_id,
        {
            "vx": 0.0,
            "vy": 0.0,
            "speed": 0.0,
            "acceleration": 0.0,
            "motion_confidence": 0.0,
        },
    )


def calculate_entity_motion_change(
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    participant_id: str,
) -> Dict[str, float]:
    before = participant_motion_at(before_state, participant_id)
    after = participant_motion_at(after_state, participant_id)

    before_velocity = (
        safe_float(before.get("vx")),
        safe_float(before.get("vy")),
    )
    after_velocity = (
        safe_float(after.get("vx")),
        safe_float(after.get("vy")),
    )

    speed_change = clamp(
        abs(
            safe_float(after.get("speed"))
            - safe_float(before.get("speed"))
        )
        / SPEED_CHANGE_SCALE
    )

    direction_change = clamp(
        angle_change(before_velocity, after_velocity)
        / DIRECTION_CHANGE_RADIANS
    )

    lateral_change = clamp(
        abs(after_velocity[0] - before_velocity[0])
        / LATERAL_CHANGE_SCALE
    )

    acceleration_change = clamp(
        abs(
            safe_float(after.get("acceleration"))
            - safe_float(before.get("acceleration"))
        )
        / ACCELERATION_SCALE
    )

    confidence = math.sqrt(
        max(safe_float(before.get("motion_confidence")), 0.05)
        * max(safe_float(after.get("motion_confidence")), 0.05)
    )

    raw_score = clamp(
        0.34 * speed_change
        + 0.28 * direction_change
        + 0.22 * lateral_change
        + 0.16 * acceleration_change
    )

    return {
        "score": clamp(raw_score * (0.55 + 0.45 * confidence)),
        "speed_change": speed_change,
        "direction_change": direction_change,
        "lateral_change": lateral_change,
        "acceleration_change": acceleration_change,
        "confidence": confidence,
    }


def calculate_pair_response(
    before_state: Dict[str, Any],
    transition_state: Dict[str, Any],
    after_state: Dict[str, Any],
) -> Dict[str, Any]:
    entity_a = transition_state["entity_a"]
    entity_b = transition_state["entity_b"]

    change_a = calculate_entity_motion_change(
        before_state,
        after_state,
        entity_a,
    )
    change_b = calculate_entity_motion_change(
        before_state,
        after_state,
        entity_b,
    )

    risk_before = safe_float(before_state.get("smoothed_risk_score"))
    risk_at_transition = safe_float(
        transition_state.get("smoothed_risk_score")
    )
    risk_after = safe_float(after_state.get("smoothed_risk_score"))

    risk_reduction = clamp(
        (risk_at_transition - risk_after)
        / max(risk_at_transition, 0.15)
    )

    miss_before = safe_float(
        transition_state.get("trajectory_proxy", {}).get(
            "predicted_min_center_distance"
        )
    )
    miss_after = safe_float(
        after_state.get("trajectory_proxy", {}).get(
            "predicted_min_center_distance"
        )
    )
    miss_improvement = clamp(
        (miss_after - miss_before)
        / max(miss_before, 0.04)
    )

    closing_before = safe_float(
        transition_state.get("trajectory_proxy", {}).get("closing_speed")
    )
    closing_after = safe_float(
        after_state.get("trajectory_proxy", {}).get("closing_speed")
    )
    closing_reduction = clamp(
        (closing_before - closing_after)
        / max(closing_before, 0.02)
    )

    # Generic motion change is only useful when pair risk geometry changes.
    pair_effect = clamp(
        0.48 * risk_reduction
        + 0.29 * miss_improvement
        + 0.23 * closing_reduction
    )

    entity_a_conditioned = clamp(change_a["score"] * pair_effect)
    entity_b_conditioned = clamp(change_b["score"] * pair_effect)

    maximum_conditioned_response = max(
        entity_a_conditioned,
        entity_b_conditioned,
    )

    trajectory_discontinuity = clamp(
        max(change_a["score"], change_b["score"])
        * (
            0.55 * max(change_a["direction_change"], change_b["direction_change"])
            + 0.45 * max(change_a["speed_change"], change_b["speed_change"])
        )
    )

    # If risk does not reduce, a strong discontinuity can still support a
    # possible impact/forced trajectory break, but at a lower weight.
    pair_response_score = clamp(
        0.78 * maximum_conditioned_response
        + 0.22 * trajectory_discontinuity
    )

    return {
        "score": pair_response_score,
        "pair_effect_score": pair_effect,
        "risk_before": risk_before,
        "risk_at_transition": risk_at_transition,
        "risk_after": risk_after,
        "risk_reduction": risk_reduction,
        "predicted_miss_improvement": miss_improvement,
        "closing_reduction": closing_reduction,
        "trajectory_discontinuity": trajectory_discontinuity,
        "entity_a_response": {
            **change_a,
            "pair_conditioned_score": entity_a_conditioned,
        },
        "entity_b_response": {
            **change_b,
            "pair_conditioned_score": entity_b_conditioned,
        },
        "response_attribution": (
            "pair_conditioned_by_risk_change_and_conflict_geometry"
        ),
    }


# ============================================================================
# JOINT RISK-TRANSITION SEARCH
# ============================================================================

def state_window(
    states: List[Dict[str, Any]],
    start_time: float,
    end_time: float,
) -> List[Dict[str, Any]]:
    return [
        state
        for state in states
        if start_time <= state["time_seconds"] <= end_time
    ]


def choose_before_state(
    states: List[Dict[str, Any]],
    transition_index: int,
) -> Optional[Dict[str, Any]]:
    transition_time = states[transition_index]["time_seconds"]
    candidates = [
        state
        for state in states[:transition_index]
        if transition_time - state["time_seconds"] <= RISK_LOOKBACK_SECONDS
    ]
    if len(candidates) < MIN_PRE_STATES:
        return None
    return candidates[0]


def choose_after_states(
    states: List[Dict[str, Any]],
    transition_index: int,
) -> List[Dict[str, Any]]:
    transition_time = states[transition_index]["time_seconds"]
    return [
        state
        for state in states[transition_index + 1:]
        if (
            transition_time + MIN_RESPONSE_DELAY_SECONDS
            < state["time_seconds"]
            <= transition_time + RESPONSE_HORIZON_SECONDS
        )
    ]


def evaluate_transition_candidate(
    states: List[Dict[str, Any]],
    transition_index: int,
) -> Optional[Dict[str, Any]]:
    transition_state = states[transition_index]
    before_state = choose_before_state(states, transition_index)
    after_states = choose_after_states(states, transition_index)

    if before_state is None or len(after_states) < MIN_POST_STATES:
        return None

    risk_before = safe_float(before_state.get("smoothed_risk_score"))
    risk_peak = safe_float(transition_state.get("smoothed_risk_score"))
    risk_escalation = clamp(
        (risk_peak - risk_before)
        / max(1.0 - risk_before, 0.20)
    )

    best = None

    for after_state in after_states:
        response = calculate_pair_response(
            before_state,
            transition_state,
            after_state,
        )

        risk_transition = clamp(
            0.60 * response["risk_reduction"]
            + 0.25 * response["predicted_miss_improvement"]
            + 0.15 * response["closing_reduction"]
        )

        topology_confidence = safe_float(
            transition_state.get("interaction_topology", {}).get("confidence")
        )
        depth_support = safe_float(
            transition_state.get("depth_compatibility", {}).get("score")
        )
        reliability = safe_float(
            transition_state.get("risk_components", {}).get(
                "motion_reliability"
            )
        )

        event_score = clamp(
            0.30 * risk_peak
            + 0.24 * risk_escalation
            + 0.22 * response["score"]
            + 0.14 * risk_transition
            + 0.05 * topology_confidence
            + 0.05 * reliability
        )

        # Depth is supporting confidence, not a hard metric-space gate.
        event_score *= 0.78 + 0.22 * depth_support
        event_score = clamp(event_score)

        evidence = {
            "transition_time": transition_state["time_seconds"],
            "response_time": after_state["time_seconds"],
            "risk_before": risk_before,
            "peak_risk": risk_peak,
            "risk_after": response["risk_after"],
            "risk_escalation": risk_escalation,
            "risk_transition_score": risk_transition,
            "pair_response": response,
            "event_score": event_score,
            "interaction_topology": transition_state[
                "interaction_topology"
            ],
            "transition_state": transition_state,
        }

        if best is None or event_score > best["event_score"]:
            best = evidence

    return best


def evaluate_risk_transition(
    states: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = []

    for transition_index in range(1, len(states) - 1):
        candidate = evaluate_transition_candidate(
            states,
            transition_index,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return {
            "candidate": False,
            "event_score": 0.0,
            "reason": "insufficient_pre_or_post_transition_evidence",
        }

    best = max(candidates, key=lambda item: item["event_score"])

    peak_risk_present = best["peak_risk"] >= MIN_PEAK_RISK
    escalation_present = best["risk_escalation"] >= MIN_RISK_ESCALATION
    response_present = (
        best["pair_response"]["score"] >= MIN_PAIR_RESPONSE
    )

    risk_transition_present = (
        best["risk_transition_score"] >= MIN_RISK_TRANSITION
    )

    discontinuity_present = (
        best["pair_response"]["trajectory_discontinuity"] >= 0.48
    )

    causal_validation = (
        response_present
        and (
            risk_transition_present
            or discontinuity_present
        )
    )

    candidate = (
        peak_risk_present
        and escalation_present
        and causal_validation
        and best["event_score"] >= MIN_EVENT_SCORE
    )

    return {
        "candidate": candidate,
        "event_score": round(best["event_score"], 4),
        "transition_time": best["transition_time"],
        "response_time": best["response_time"],
        "risk_before": round(best["risk_before"], 4),
        "peak_risk": round(best["peak_risk"], 4),
        "risk_after": round(best["risk_after"], 4),
        "risk_escalation": round(best["risk_escalation"], 4),
        "risk_transition_score": round(
            best["risk_transition_score"], 4
        ),
        "pair_response": round(best["pair_response"]["score"], 4),
        "trajectory_discontinuity": round(
            best["pair_response"]["trajectory_discontinuity"], 4
        ),
        "interaction_topology": best["interaction_topology"],
        "validation": {
            "peak_risk_present": peak_risk_present,
            "risk_escalation_present": escalation_present,
            "pair_conditioned_response_present": response_present,
            "risk_transition_present": risk_transition_present,
            "trajectory_discontinuity_present": discontinuity_present,
            "causal_validation": causal_validation,
        },
        "response_evidence": serialize_response(
            best["pair_response"]
        ),
        "transition_risk_components": best[
            "transition_state"
        ]["risk_components"],
        "trajectory_proxy": best[
            "transition_state"
        ]["trajectory_proxy"],
    }


def serialize_response(response: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in response.items():
        if isinstance(value, dict):
            result[key] = {
                subkey: round(subvalue, 4)
                if isinstance(subvalue, float)
                else subvalue
                for subkey, subvalue in value.items()
            }
        elif isinstance(value, float):
            result[key] = round(value, 4)
        else:
            result[key] = value
    return result


# ============================================================================
# EPISODES
# ============================================================================

def dominant_topology(states: List[Dict[str, Any]]) -> str:
    weighted: Dict[str, float] = defaultdict(float)
    for state in states:
        topology = state.get("interaction_topology", {})
        weighted[str(topology.get("type", "UNKNOWN"))] += (
            safe_float(topology.get("confidence"))
            * max(safe_float(state.get("smoothed_risk_score")), 0.05)
        )
    return max(weighted, key=weighted.get) if weighted else "UNKNOWN"


def build_interaction_episodes(
    timeline: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pair_states = collect_pair_states(timeline)
    episodes = []

    for states in pair_states.values():
        for episode_states in split_into_episodes(states):
            transition = evaluate_risk_transition(episode_states)

            peak_state = max(
                episode_states,
                key=lambda item: safe_float(
                    item.get("smoothed_risk_score")
                ),
            )

            episode = {
                "entity_a": episode_states[0]["entity_a"],
                "entity_b": episode_states[0]["entity_b"],
                "visual_identity_a": episode_states[0][
                    "visual_identity_a"
                ],
                "visual_identity_b": episode_states[0][
                    "visual_identity_b"
                ],
                "class_a": episode_states[0]["class_a"],
                "class_b": episode_states[0]["class_b"],
                "start_time": episode_states[0]["time_seconds"],
                "end_time": episode_states[-1]["time_seconds"],
                "duration_seconds": round(
                    episode_states[-1]["time_seconds"]
                    - episode_states[0]["time_seconds"],
                    3,
                ),
                "states": len(episode_states),
                "dominant_interaction_topology": dominant_topology(
                    episode_states
                ),
                "peak_risk": round(
                    safe_float(peak_state.get("smoothed_risk_score")),
                    4,
                ),
                "peak_risk_time": peak_state["time_seconds"],
                "average_risk": round(
                    mean_or_zero(
                        [
                            safe_float(
                                state.get("smoothed_risk_score")
                            )
                            for state in episode_states
                        ]
                    ),
                    4,
                ),
                "minimum_edge_distance": round(
                    min(
                        state["geometry"]["edge_distance"]
                        for state in episode_states
                    ),
                    6,
                ),
                "maximum_bbox_overlap_ratio": round(
                    max(
                        state["geometry"]["bbox_overlap_ratio"]
                        for state in episode_states
                    ),
                    4,
                ),
                "risk_transition": transition,
                "critical_event_candidate": transition["candidate"],
                "critical_event_score": transition["event_score"],
                "interpretation": (
                    "Pairwise conflict-risk episode. Critical-event "
                    "eligibility requires risk escalation plus a later "
                    "pair-conditioned response that changes this pair's "
                    "risk geometry, or a strong trajectory discontinuity. "
                    "This is not collision confirmation."
                ),
            }
            episodes.append(episode)

    episodes.sort(
        key=lambda item: item["critical_event_score"],
        reverse=True,
    )

    for index, episode in enumerate(episodes, start=1):
        episode["interaction_episode_id"] = f"Interaction_{index}"

    return episodes


# ============================================================================
# CANDIDATE DEDUPLICATION / COMPETITION
# ============================================================================

def events_overlap(
    event_a: Dict[str, Any],
    event_b: Dict[str, Any],
) -> bool:
    return abs(
        safe_float(event_a["transition_time"])
        - safe_float(event_b["transition_time"])
    ) <= DUPLICATE_EVENT_TIME_WINDOW


def participant_competition(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    If the same participant has many candidate pairs in the same short window,
    keep only its strongest few pair explanations. This prevents a single
    braking/direction-change event from propagating to every nearby pair.
    """
    accepted = []

    for candidate in sorted(
        candidates,
        key=lambda item: item["critical_event_score"],
        reverse=True,
    ):
        reject = False

        for participant in (
            candidate["entity_a"],
            candidate["entity_b"],
        ):
            competing = [
                item
                for item in accepted
                if participant in {
                    item["entity_a"],
                    item["entity_b"],
                }
                and events_overlap(candidate, item)
            ]

            if len(competing) >= MAX_CRITICAL_PAIRS_PER_ENTITY_WINDOW:
                reject = True
                break

        if not reject:
            accepted.append(candidate)

    return accepted


def build_critical_event_candidates(
    episodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates = []

    for episode in episodes:
        transition = episode["risk_transition"]

        if not transition.get("candidate", False):
            continue

        candidates.append(
            {
                "entity_a": episode["entity_a"],
                "entity_b": episode["entity_b"],
                "visual_identity_a": episode["visual_identity_a"],
                "visual_identity_b": episode["visual_identity_b"],
                "class_a": episode["class_a"],
                "class_b": episode["class_b"],
                "interaction_episode_id": episode[
                    "interaction_episode_id"
                ],
                "event_start_time": episode["start_time"],
                "transition_time": transition["transition_time"],
                "response_time": transition["response_time"],
                "event_end_time": episode["end_time"],
                "interaction_topology": transition[
                    "interaction_topology"
                ],
                "risk_before": transition["risk_before"],
                "peak_risk": transition["peak_risk"],
                "risk_after": transition["risk_after"],
                "risk_escalation": transition["risk_escalation"],
                "risk_transition_score": transition[
                    "risk_transition_score"
                ],
                "pair_conditioned_response_score": transition[
                    "pair_response"
                ],
                "trajectory_discontinuity": transition[
                    "trajectory_discontinuity"
                ],
                "critical_event_score": episode[
                    "critical_event_score"
                ],
                "validation": transition["validation"],
                "response_evidence": transition["response_evidence"],
                "transition_risk_components": transition[
                    "transition_risk_components"
                ],
                "trajectory_proxy": transition["trajectory_proxy"],
                "evidence_statement": (
                    "Critical-event candidate because pairwise risk "
                    "escalated and a later participant response was "
                    "validated against the risk evolution of this exact "
                    "pair. Generic motion change alone is insufficient. "
                    "This is not collision confirmation."
                ),
            }
        )

    candidates = participant_competition(candidates)

    candidates.sort(
        key=lambda item: item["critical_event_score"],
        reverse=True,
    )

    for index, candidate in enumerate(candidates, start=1):
        candidate["critical_event_id"] = f"CriticalEvent_{index}"

    return candidates


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_interactions(data: Dict[str, Any]) -> Dict[str, Any]:
    timeline = build_interaction_timeline(data)
    episodes = build_interaction_episodes(timeline)
    critical_events = build_critical_event_candidates(episodes)

    return {
        "configuration": {
            "version": "v6",
            "standalone": True,
            "pairwise_conflict_risk_timeline": True,
            "interaction_topology": True,
            "joint_risk_transition_search": True,
            "pair_conditioned_response_attribution": True,
            "risk_reduction_validation": True,
            "trajectory_discontinuity_fallback": True,
            "image_space_metrics_are_proxies": True,
            "bbox_overlap_is_collision": False,
            "accident_classification": False,
            "pre_accident_alerting": False,
            "design_note": (
                "V6 replaces trigger-response matching with pairwise "
                "risk-transition validation. A critical candidate requires "
                "elevated pair risk, risk escalation, and a later response "
                "whose effect is validated against that exact pair's risk "
                "geometry, or a strong trajectory discontinuity."
            ),
        },
        "interaction_timeline": timeline,
        "interaction_episodes": episodes,
        "critical_event_candidates": critical_events,
        "summary": {
            "timeline_states": len(timeline),
            "interaction_episodes": len(episodes),
            "pairs_with_episodes": len(
                {
                    pair_key(
                        episode["entity_a"],
                        episode["entity_b"],
                    )
                    for episode in episodes
                }
            ),
            "critical_event_candidates": len(critical_events),
        },
    }


# ============================================================================
# OUTPUT
# ============================================================================

def derive_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_object_analysis"):
        stem = stem[:-len("_object_analysis")]
    return input_path.with_name(f"{stem}_interaction_analysis.json")


def print_summary(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 92)
    print("INTERACTION ANALYZER V6 COMPLETE")
    print("=" * 92)

    print(
        f"\nInteraction episodes      : "
        f"{result['summary']['interaction_episodes']}"
    )
    print(
        f"Pairs with episodes       : "
        f"{result['summary']['pairs_with_episodes']}"
    )
    print(
        f"Critical event candidates : "
        f"{result['summary']['critical_event_candidates']}"
    )

    candidates = result["critical_event_candidates"]

    if not candidates:
        print(
            "\nNo pairwise risk-transition critical event "
            "was confirmed."
        )
    else:
        print("\nCritical event candidates:")

        for candidate in candidates[:20]:
            print(
                f"\n{candidate['critical_event_id']}  "
                f"{candidate['visual_identity_a']} <-> "
                f"{candidate['visual_identity_b']}"
            )
            print(
                f"  Classes       : "
                f"{candidate['class_a']} <-> {candidate['class_b']}"
            )
            print(
                f"  Topology      : "
                f"{candidate['interaction_topology']['type']}"
            )
            print(
                f"  Event window  : "
                f"{candidate['event_start_time']:.2f} - "
                f"{candidate['event_end_time']:.2f} s"
            )
            print(
                f"  Transition    : "
                f"{candidate['transition_time']:.2f} s"
            )
            print(
                f"  Response      : "
                f"{candidate['response_time']:.2f} s"
            )
            print(
                f"  Risk          : "
                f"{candidate['risk_before']:.3f} -> "
                f"{candidate['peak_risk']:.3f} -> "
                f"{candidate['risk_after']:.3f}"
            )
            print(
                f"  Escalation    : "
                f"{candidate['risk_escalation']:.4f}"
            )
            print(
                f"  Pair response : "
                f"{candidate['pair_conditioned_response_score']:.4f}"
            )
            print(
                f"  Risk change   : "
                f"{candidate['risk_transition_score']:.4f}"
            )
            print(
                f"  Event score   : "
                f"{candidate['critical_event_score']:.4f}"
            )

    print("\n" + "=" * 92)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone V6 pairwise conflict-risk and "
            "risk-transition analyzer."
        )
    )

    parser.add_argument(
        "object_analysis_json",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    print("[interaction] Loading object analysis...")
    data = load_json(args.object_analysis_json)

    print("[interaction] Building pairwise conflict-risk timelines...")
    result = analyze_interactions(data)

    output_path = (
        args.output
        if args.output is not None
        else derive_output_path(args.object_analysis_json)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print_summary(result)

    print(f"\nJSON written to: {output_path}")


if __name__ == "__main__":
    main()
