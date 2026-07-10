"""
interaction_analyzer_v5.py

Standalone Stage-3 V5 interaction and critical-event analyzer.

V5 philosophy
--------------
An interaction is NOT a critical event.

Bounding-box overlap, edge distance, proximity, convergence, and predicted
closest approach describe pairwise traffic geometry. They are useful for
identifying a possible trigger, but they are NOT accident evidence by
themselves.

A critical-event candidate requires a temporal event chain:

    PRE-INTERACTION
        normal or developing pair relationship

            ->

    TRIGGER
        rapid distance compression
        strong closing motion
        path conflict
        sudden intrusion
        vulnerable-road-user exposure

            ->

    RESPONSE
        rapid-to-low motion
        direction change
        lateral motion onset
        abrupt motion-state change
        strong acceleration change
        post-trigger separation anomaly
        short post-trigger track disappearance

The trigger and response must be temporally linked.

Input:
    *_object_analysis.json

Output:
    *_interaction_analysis.json

This stage:
    - analyzes pairwise interactions
    - builds interaction episodes
    - detects trigger-response event chains
    - returns critical event candidates

This stage DOES NOT:
    - confirm collision
    - classify an accident
    - infer injury
    - infer fault
    - issue pre-accident alerts
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
# SUPPORTED ROAD USERS
# ============================================================================

SUPPORTED_ROAD_USERS = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "motorbike",
    "bicycle",
    "bike",
    "person",
    "pedestrian",
}

VULNERABLE_ROAD_USERS = {
    "motorcycle",
    "motorbike",
    "bicycle",
    "bike",
    "person",
    "pedestrian",
}


# ============================================================================
# PAIR ANALYSIS CONFIGURATION
# ============================================================================

MAX_PAIR_CENTER_DISTANCE = 0.48

DEPTH_CENTER_Y_TOLERANCE = 0.18
DEPTH_COMPATIBILITY_MIN = 0.42

MAX_LOG_AREA_RATIO_VEHICLE_PAIR = 1.55
MAX_LOG_AREA_RATIO_VRU_PAIR = 2.75

NEAR_EDGE_DISTANCE = 0.12

CONVERGENCE_SPEED_SCALE = 0.10
MIN_CLOSING_SPEED = 0.008
STRONG_CLOSING_SPEED = 0.025

PREDICTION_HORIZON_SECONDS = 1.20
MIN_RELATIVE_SPEED = 0.010

CONFLICT_DISTANCE = 0.070
STRONG_CONFLICT_DISTANCE = 0.040

INTERACTION_ACTIVE_THRESHOLD = 0.18


# ============================================================================
# TEMPORAL CONFIGURATION
# ============================================================================

PAIR_EVENT_LOOKBACK_SECONDS = 0.65

EPISODE_MIN_STATES = 2
EPISODE_MAX_GAP_SECONDS = 0.21
EPISODE_MIN_SCORE = 0.16

PRE_TRIGGER_WINDOW_SECONDS = 0.60
RESPONSE_WINDOW_SECONDS = 0.80

TRACK_DISAPPEARANCE_WINDOW_SECONDS = 0.45

MIN_TRIGGER_SCORE = 0.38
MIN_RESPONSE_SCORE = 0.38

CRITICAL_CHAIN_THRESHOLD = 0.52

SHORT_EVENT_SECONDS = 1.20

# V5: trigger-centred, pair-specific response attribution
LOCAL_EVENT_PRE_TRIGGER_SECONDS = 0.12
LOCAL_EVENT_POST_TRIGGER_SECONDS = 0.80
CONTINUOUS_BASELINE_SECONDS = 0.35
CONTINUOUS_SPEED_CHANGE_SCALE = 0.06
CONTINUOUS_DIRECTION_CHANGE_DEGREES = 55.0
CONTINUOUS_LATERAL_CHANGE_SCALE = 0.045


# ============================================================================
# MOTION EVENT DEFINITIONS
# ============================================================================

RESPONSE_EVENTS = {
    "rapid_to_low_motion",
    "direction_change",
    "lateral_motion_onset",
    "low_to_rapid_motion",
}

STRONG_RESPONSE_EVENTS = {
    "rapid_to_low_motion",
    "direction_change",
    "lateral_motion_onset",
}


# ============================================================================
# BASIC UTILITIES
# ============================================================================


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(
        low,
        min(
            high,
            value,
        ),
    )


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def mean_or_zero(
    values: List[float],
) -> float:
    if not values:
        return 0.0

    return float(
        np.mean(
            values
        )
    )


def load_json(
    path: Path,
) -> Dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def pair_key(
    entity_a: str,
    entity_b: str,
) -> Tuple[str, str]:
    return tuple(
        sorted(
            (
                entity_a,
                entity_b,
            )
        )
    )


def bbox_edge_distance(
    box_a: List[float],
    box_b: List[float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    dx = max(
        bx1 - ax2,
        ax1 - bx2,
        0.0,
    )

    dy = max(
        by1 - ay2,
        ay1 - by2,
        0.0,
    )

    return math.hypot(
        dx,
        dy,
    )


def bbox_overlap_ratio(
    box_a: List[float],
    box_b: List[float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(
        ax1,
        bx1,
    )

    y1 = max(
        ay1,
        by1,
    )

    x2 = min(
        ax2,
        bx2,
    )

    y2 = min(
        ay2,
        by2,
    )

    intersection = (
        max(
            0.0,
            x2 - x1,
        )
        *
        max(
            0.0,
            y2 - y1,
        )
    )

    area_a = max(
        (
            ax2 - ax1
        )
        *
        (
            ay2 - ay1
        ),
        1e-8,
    )

    area_b = max(
        (
            bx2 - bx1
        )
        *
        (
            by2 - by1
        ),
        1e-8,
    )

    return intersection / min(
        area_a,
        area_b,
    )


def is_supported_entity(
    state: Dict[str, Any],
) -> bool:
    class_name = str(
        state.get(
            "class_name",
            "",
        )
    ).lower()

    return (
        class_name
        in SUPPORTED_ROAD_USERS
    )


def is_vulnerable_class(
    class_name: str,
) -> bool:
    return (
        str(
            class_name
        ).lower()
        in VULNERABLE_ROAD_USERS
    )


def pair_has_vru(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
) -> bool:
    return (
        is_vulnerable_class(
            state_a.get(
                "class_name",
                "",
            )
        )
        or
        is_vulnerable_class(
            state_b.get(
                "class_name",
                "",
            )
        )
    )


# ============================================================================
# INPUT INDEXING
# ============================================================================


def build_time_index(
    data: Dict[str, Any],
) -> Dict[
    float,
    List[Dict[str, Any]],
]:
    result = defaultdict(
        list
    )

    for states in data.get(
        "entity_states",
        {},
    ).values():

        if not isinstance(
            states,
            list,
        ):
            continue

        for state in states:

            if not isinstance(
                state,
                dict,
            ):
                continue

            if not is_supported_entity(
                state
            ):
                continue

            time_seconds = round(
                safe_float(
                    state.get(
                        "time_seconds"
                    )
                ),
                3,
            )

            result[
                time_seconds
            ].append(
                state
            )

    return dict(
        sorted(
            result.items()
        )
    )


def build_entity_history(
    data: Dict[str, Any],
) -> Dict[
    str,
    List[Dict[str, Any]],
]:
    result = {}

    for entity_id, states in data.get(
        "entity_states",
        {},
    ).items():

        if not isinstance(
            states,
            list,
        ):
            continue

        result[
            str(
                entity_id
            )
        ] = sorted(
            [
                state
                for state in states
                if isinstance(
                    state,
                    dict,
                )
            ],
            key=lambda state: safe_float(
                state.get(
                    "time_seconds"
                )
            ),
        )

    return result


# ============================================================================
# MOTION RELIABILITY
# ============================================================================


def reliability_weight(
    state: Dict[str, Any],
) -> float:
    quality = state.get(
        "motion_quality",
        {},
    )

    confidence = clamp(
        safe_float(
            quality.get(
                "motion_confidence"
            )
        )
    )

    reliability = quality.get(
        "reliability",
        "low",
    )

    multiplier = {
        "low": 0.35,
        "moderate": 0.70,
        "high": 1.00,
    }.get(
        reliability,
        0.35,
    )

    return clamp(
        confidence
        * multiplier
    )


# ============================================================================
# DEPTH COMPATIBILITY
# ============================================================================


def calculate_depth_compatibility(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
) -> Dict[str, Any]:
    geometry_a = state_a[
        "geometry"
    ]

    geometry_b = state_b[
        "geometry"
    ]

    center_y_gap = abs(
        safe_float(
            geometry_a[
                "center_y"
            ]
        )
        -
        safe_float(
            geometry_b[
                "center_y"
            ]
        )
    )

    area_a = max(
        safe_float(
            geometry_a[
                "area"
            ]
        ),
        1e-8,
    )

    area_b = max(
        safe_float(
            geometry_b[
                "area"
            ]
        ),
        1e-8,
    )

    log_area_ratio = abs(
        math.log(
            area_a
            / area_b
        )
    )

    vulnerable_pair = (
        pair_has_vru(
            state_a,
            state_b,
        )
    )

    maximum_scale_gap = (
        MAX_LOG_AREA_RATIO_VRU_PAIR
        if vulnerable_pair
        else MAX_LOG_AREA_RATIO_VEHICLE_PAIR
    )

    scale_gate_passed = (
        log_area_ratio
        <= maximum_scale_gap
    )

    vertical_score = clamp(
        1.0
        -
        center_y_gap
        / DEPTH_CENTER_Y_TOLERANCE
    )

    scale_score = clamp(
        1.0
        -
        log_area_ratio
        / max(
            maximum_scale_gap,
            1e-8,
        )
    )

    depth_a = state_a.get(
        "depth_state",
        "uncertain",
    )

    depth_b = state_b.get(
        "depth_state",
        "uncertain",
    )

    same_depth_trend = (
        depth_a == depth_b
        and depth_a
        in {
            "approaching",
            "receding",
            "stable",
        }
    )

    trend_score = (
        1.0
        if same_depth_trend
        else 0.45
        if "uncertain"
        in {
            depth_a,
            depth_b,
        }
        else 0.20
    )

    compatibility = clamp(
        0.50
        * vertical_score
        +
        0.35
        * scale_score
        +
        0.15
        * trend_score
    )

    if not scale_gate_passed:
        compatibility *= 0.20

    return {
        "score": compatibility,
        "compatible": (
            scale_gate_passed
            and compatibility
            >= DEPTH_COMPATIBILITY_MIN
        ),
        "center_y_gap": center_y_gap,
        "log_area_ratio": log_area_ratio,
        "scale_gate_passed": (
            scale_gate_passed
        ),
        "vulnerable_road_user_pair": (
            vulnerable_pair
        ),
    }


# ============================================================================
# TRAJECTORY ANALYSIS
# ============================================================================


def calculate_trajectory_state(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
) -> Dict[str, Any]:
    geometry_a = state_a[
        "geometry"
    ]

    geometry_b = state_b[
        "geometry"
    ]

    motion_a = state_a[
        "relative_motion"
    ]

    motion_b = state_b[
        "relative_motion"
    ]

    rx = (
        safe_float(
            geometry_b[
                "center_x"
            ]
        )
        -
        safe_float(
            geometry_a[
                "center_x"
            ]
        )
    )

    ry = (
        safe_float(
            geometry_b[
                "center_y"
            ]
        )
        -
        safe_float(
            geometry_a[
                "center_y"
            ]
        )
    )

    rvx = (
        safe_float(
            motion_b[
                "vx"
            ]
        )
        -
        safe_float(
            motion_a[
                "vx"
            ]
        )
    )

    rvy = (
        safe_float(
            motion_b[
                "vy"
            ]
        )
        -
        safe_float(
            motion_a[
                "vy"
            ]
        )
    )

    center_distance = math.hypot(
        rx,
        ry,
    )

    relative_speed = math.hypot(
        rvx,
        rvy,
    )

    position_velocity_dot = (
        rx * rvx
        +
        ry * rvy
    )

    if center_distance > 1e-8:

        closing_speed = max(
            0.0,
            -position_velocity_dot
            / center_distance,
        )

    else:

        closing_speed = (
            relative_speed
        )

    converging = (
        position_velocity_dot < 0.0
        and closing_speed
        >= MIN_CLOSING_SPEED
    )

    convergence_score = clamp(
        closing_speed
        / CONVERGENCE_SPEED_SCALE
    )

    predicted_min_distance = (
        center_distance
    )

    time_to_closest = None
    future_conflict_score = 0.0

    relative_speed_sq = (
        rvx * rvx
        +
        rvy * rvy
    )

    if (
        converging
        and relative_speed_sq
        >= MIN_RELATIVE_SPEED ** 2
    ):

        candidate_time = (
            -position_velocity_dot
            / relative_speed_sq
        )

        if (
            0.0
            < candidate_time
            <= PREDICTION_HORIZON_SECONDS
        ):

            time_to_closest = (
                candidate_time
            )

            predicted_rx = (
                rx
                +
                rvx
                * candidate_time
            )

            predicted_ry = (
                ry
                +
                rvy
                * candidate_time
            )

            predicted_min_distance = (
                math.hypot(
                    predicted_rx,
                    predicted_ry,
                )
            )

            future_conflict_score = clamp(
                (
                    CONFLICT_DISTANCE
                    -
                    predicted_min_distance
                )
                /
                max(
                    CONFLICT_DISTANCE
                    -
                    STRONG_CONFLICT_DISTANCE,
                    1e-8,
                )
            )

    return {
        "center_distance": (
            center_distance
        ),
        "relative_speed": (
            relative_speed
        ),
        "closing_speed": (
            closing_speed
        ),
        "convergence_score": (
            convergence_score
        ),
        "converging": (
            converging
        ),
        "time_to_closest": (
            time_to_closest
        ),
        "predicted_min_distance": (
            predicted_min_distance
        ),
        "future_conflict_score": (
            future_conflict_score
        ),
    }


# ============================================================================
# RECENT MOTION EVENTS
# ============================================================================


def recent_motion_events(
    history: Dict[
        str,
        List[Dict[str, Any]],
    ],
    entity_id: str,
    current_time: float,
) -> List[Dict[str, Any]]:
    events = []

    start_time = (
        current_time
        -
        PAIR_EVENT_LOOKBACK_SECONDS
    )

    for state in history.get(
        entity_id,
        [],
    ):

        state_time = safe_float(
            state.get(
                "time_seconds"
            )
        )

        if state_time < start_time:
            continue

        if state_time > current_time:
            break

        for event in state.get(
            "motion_events",
            [],
        ):

            if not isinstance(
                event,
                dict,
            ):
                continue

            event_time = safe_float(
                event.get(
                    "time_seconds",
                    state_time,
                )
            )

            if not (
                start_time
                <= event_time
                <= current_time
            ):
                continue

            events.append(
                {
                    "entity_id": (
                        entity_id
                    ),
                    "event": event.get(
                        "event",
                        "unknown",
                    ),
                    "time_seconds": round(
                        event_time,
                        3,
                    ),
                    "confidence": round(
                        clamp(
                            safe_float(
                                event.get(
                                    "confidence"
                                )
                            )
                        ),
                        4,
                    ),
                }
            )

    return events


# ============================================================================
# PAIR STATE
# ============================================================================


def analyze_pair_state(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
    history: Dict[
        str,
        List[Dict[str, Any]],
    ],
    current_time: float,
) -> Optional[Dict[str, Any]]:
    geometry_a = state_a[
        "geometry"
    ]

    geometry_b = state_b[
        "geometry"
    ]

    trajectory = (
        calculate_trajectory_state(
            state_a,
            state_b,
        )
    )

    if (
        trajectory[
            "center_distance"
        ]
        > MAX_PAIR_CENTER_DISTANCE
    ):
        return None

    edge_distance = (
        bbox_edge_distance(
            geometry_a[
                "bbox"
            ],
            geometry_b[
                "bbox"
            ],
        )
    )

    overlap_ratio = (
        bbox_overlap_ratio(
            geometry_a[
                "bbox"
            ],
            geometry_b[
                "bbox"
            ],
        )
    )

    depth = (
        calculate_depth_compatibility(
            state_a,
            state_b,
        )
    )

    reliability = math.sqrt(
        reliability_weight(
            state_a
        )
        *
        reliability_weight(
            state_b
        )
    )

    proximity_score = clamp(
        (
            NEAR_EDGE_DISTANCE
            -
            edge_distance
        )
        /
        NEAR_EDGE_DISTANCE
    )

    interaction_score = clamp(
        (
            0.30
            * proximity_score
            +
            0.30
            * trajectory[
                "convergence_score"
            ]
            +
            0.25
            * trajectory[
                "future_conflict_score"
            ]
            * depth[
                "score"
            ]
            +
            0.15
            * depth[
                "score"
            ]
        )
        *
        (
            0.45
            +
            0.55
            * reliability
        )
    )

    if (
        interaction_score
        < INTERACTION_ACTIVE_THRESHOLD
        and trajectory[
            "closing_speed"
        ]
        < MIN_CLOSING_SPEED
    ):
        return None

    events = (
        recent_motion_events(
            history,
            state_a[
                "entity_id"
            ],
            current_time,
        )
        +
        recent_motion_events(
            history,
            state_b[
                "entity_id"
            ],
            current_time,
        )
    )

    if (
        trajectory[
            "future_conflict_score"
        ]
        >= 0.50
        and depth[
            "compatible"
        ]
    ):

        interaction_type = (
            "trajectory_conflict"
        )

    elif (
        trajectory[
            "convergence_score"
        ]
        >= 0.55
        and depth[
            "compatible"
        ]
    ):

        interaction_type = (
            "trajectory_convergence"
        )

    elif (
        edge_distance
        <= NEAR_EDGE_DISTANCE
    ):

        interaction_type = (
            "proximity"
        )

    else:

        interaction_type = (
            "pairwise_motion"
        )

    return {
        "time_seconds": round(
            current_time,
            3,
        ),
        "entity_a": state_a[
            "entity_id"
        ],
        "entity_b": state_b[
            "entity_id"
        ],
        "visual_identity_a": (
            state_a.get(
                "visual_identity",
                state_a[
                    "entity_id"
                ],
            )
        ),
        "visual_identity_b": (
            state_b.get(
                "visual_identity",
                state_b[
                    "entity_id"
                ],
            )
        ),
        "class_a": state_a.get(
            "class_name",
            "object",
        ),
        "class_b": state_b.get(
            "class_name",
            "object",
        ),
        "interaction_type": (
            interaction_type
        ),
        "interaction_score": round(
            interaction_score,
            4,
        ),
        "pair_reliability": round(
            reliability,
            4,
        ),
        "geometry": {
            "center_distance": round(
                trajectory[
                    "center_distance"
                ],
                6,
            ),
            "edge_distance": round(
                edge_distance,
                6,
            ),
            "bbox_overlap_ratio": round(
                overlap_ratio,
                4,
            ),
            "critical_event_weight": 0.0,
            "note": (
                "Geometry is interaction context only. "
                "Overlap and proximity cannot directly "
                "create a critical event."
            ),
        },
        "depth_compatibility": {
            "compatible": (
                depth[
                    "compatible"
                ]
            ),
            "score": round(
                depth[
                    "score"
                ],
                4,
            ),
            "center_y_gap": round(
                depth[
                    "center_y_gap"
                ],
                6,
            ),
            "log_area_ratio": round(
                depth[
                    "log_area_ratio"
                ],
                6,
            ),
            "scale_gate_passed": (
                depth[
                    "scale_gate_passed"
                ]
            ),
            "vulnerable_road_user_pair": (
                depth[
                    "vulnerable_road_user_pair"
                ]
            ),
        },
        "trajectory": {
            "relative_speed": round(
                trajectory[
                    "relative_speed"
                ],
                6,
            ),
            "closing_speed": round(
                trajectory[
                    "closing_speed"
                ],
                6,
            ),
            "convergence_score": round(
                trajectory[
                    "convergence_score"
                ],
                4,
            ),
            "converging": (
                trajectory[
                    "converging"
                ]
            ),
            "time_to_closest_proxy": (
                round(
                    trajectory[
                        "time_to_closest"
                    ],
                    4,
                )
                if trajectory[
                    "time_to_closest"
                ]
                is not None
                else None
            ),
            "predicted_min_center_distance": round(
                trajectory[
                    "predicted_min_distance"
                ],
                6,
            ),
            "future_conflict_score": round(
                trajectory[
                    "future_conflict_score"
                ],
                4,
            ),
        },
        "motion_events": (
            events
        ),
        "ego_path_context": {
            "entity_a_inside": (
                state_a.get(
                    "ego_path",
                    {},
                ).get(
                    "currently_inside",
                    False,
                )
            ),
            "entity_b_inside": (
                state_b.get(
                    "ego_path",
                    {},
                ).get(
                    "currently_inside",
                    False,
                )
            ),
            "entity_a_moving_toward": (
                state_a.get(
                    "ego_path",
                    {},
                ).get(
                    "moving_toward_corridor",
                    False,
                )
            ),
            "entity_b_moving_toward": (
                state_b.get(
                    "ego_path",
                    {},
                ).get(
                    "moving_toward_corridor",
                    False,
                )
            ),
        },
    }


# ============================================================================
# INTERACTION TIMELINE
# ============================================================================


def build_interaction_timeline(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    time_index = (
        build_time_index(
            data
        )
    )

    history = (
        build_entity_history(
            data
        )
    )

    timeline = []

    for (
        current_time,
        states,
    ) in time_index.items():

        interactions = []

        for index_a in range(
            len(
                states
            )
        ):

            for index_b in range(
                index_a + 1,
                len(
                    states
                ),
            ):

                interaction = (
                    analyze_pair_state(
                        states[
                            index_a
                        ],
                        states[
                            index_b
                        ],
                        history,
                        current_time,
                    )
                )

                if interaction is not None:

                    interactions.append(
                        interaction
                    )

        interactions.sort(
            key=lambda item: item[
                "interaction_score"
            ],
            reverse=True,
        )

        timeline.append(
            {
                "time_seconds": (
                    current_time
                ),
                "active_interactions": (
                    interactions
                ),
            }
        )

    return timeline


# ============================================================================
# EPISODE CONSTRUCTION
# ============================================================================


def collect_pair_states(
    timeline: List[Dict[str, Any]],
) -> Dict[
    Tuple[str, str],
    List[Dict[str, Any]],
]:
    result = defaultdict(
        list
    )

    for timeline_state in timeline:

        for interaction in timeline_state[
            "active_interactions"
        ]:

            key = pair_key(
                interaction[
                    "entity_a"
                ],
                interaction[
                    "entity_b"
                ],
            )

            result[
                key
            ].append(
                interaction
            )

    for states in result.values():

        states.sort(
            key=lambda state: state[
                "time_seconds"
            ]
        )

    return result


def split_into_episodes(
    states: List[Dict[str, Any]],
) -> List[
    List[Dict[str, Any]]
]:
    if not states:
        return []

    episodes = []

    current = [
        states[0]
    ]

    for state in states[1:]:

        gap = (
            state[
                "time_seconds"
            ]
            -
            current[-1][
                "time_seconds"
            ]
        )

        if (
            gap
            <= EPISODE_MAX_GAP_SECONDS
        ):

            current.append(
                state
            )

        else:

            if (
                len(
                    current
                )
                >= EPISODE_MIN_STATES
                and max(
                    item[
                        "interaction_score"
                    ]
                    for item in current
                )
                >= EPISODE_MIN_SCORE
            ):

                episodes.append(
                    current
                )

            current = [
                state
            ]

    if (
        len(
            current
        )
        >= EPISODE_MIN_STATES
        and max(
            item[
                "interaction_score"
            ]
            for item in current
        )
        >= EPISODE_MIN_SCORE
    ):

        episodes.append(
            current
        )

    return episodes


# ============================================================================
# TRIGGER ANALYSIS
# ============================================================================


def calculate_trigger_evidence(
    states: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not states:

        return {
            "score": 0.0,
            "time_seconds": None,
            "trigger_type": "none",
            "evidence": {},
        }

    candidates = []

    for index, state in enumerate(
        states
    ):

        current_distance = state[
            "geometry"
        ][
            "center_distance"
        ]

        previous_states = [
            previous
            for previous in states
            if (
                0.0
                <
                state[
                    "time_seconds"
                ]
                -
                previous[
                    "time_seconds"
                ]
                <= PRE_TRIGGER_WINDOW_SECONDS
            )
        ]

        distance_compression = 0.0
        interaction_growth = 0.0

        if previous_states:

            previous = (
                previous_states[0]
            )

            previous_distance = (
                previous[
                    "geometry"
                ][
                    "center_distance"
                ]
            )

            distance_compression = clamp(
                (
                    previous_distance
                    -
                    current_distance
                )
                /
                max(
                    previous_distance,
                    0.03,
                )
            )

            interaction_growth = clamp(
                (
                    state[
                        "interaction_score"
                    ]
                    -
                    previous[
                        "interaction_score"
                    ]
                )
                /
                0.35
            )

        closing_score = clamp(
            state[
                "trajectory"
            ][
                "closing_speed"
            ]
            /
            STRONG_CLOSING_SPEED
        )

        path_conflict_score = (
            state[
                "trajectory"
            ][
                "future_conflict_score"
            ]
            if state[
                "depth_compatibility"
            ][
                "compatible"
            ]
            else 0.0
        )

        ego_intrusion_score = 0.0

        ego_context = state[
            "ego_path_context"
        ]

        if (
            ego_context[
                "entity_a_moving_toward"
            ]
            or ego_context[
                "entity_b_moving_toward"
            ]
        ):

            ego_intrusion_score = 1.0

        elif (
            ego_context[
                "entity_a_inside"
            ]
            or ego_context[
                "entity_b_inside"
            ]
        ):

            ego_intrusion_score = 0.45

        vulnerable_exposure = 0.0

        if state[
            "depth_compatibility"
        ][
            "vulnerable_road_user_pair"
        ]:

            vulnerable_exposure = clamp(
                0.45
                * closing_score
                +
                0.35
                * path_conflict_score
                +
                0.20
                * distance_compression
            )

        trigger_score = clamp(
            0.25
            * distance_compression
            +
            0.20
            * interaction_growth
            +
            0.20
            * closing_score
            +
            0.20
            * path_conflict_score
            +
            0.10
            * ego_intrusion_score
            +
            0.05
            * vulnerable_exposure
        )

        trigger_components = {
            "distance_compression": round(
                distance_compression,
                4,
            ),
            "interaction_growth": round(
                interaction_growth,
                4,
            ),
            "closing_motion": round(
                closing_score,
                4,
            ),
            "path_conflict": round(
                path_conflict_score,
                4,
            ),
            "ego_path_intrusion": round(
                ego_intrusion_score,
                4,
            ),
            "vulnerable_exposure": round(
                vulnerable_exposure,
                4,
            ),
        }

        strongest_component = max(
            trigger_components,
            key=trigger_components.get,
        )

        candidates.append(
            {
                "score": trigger_score,
                "time_seconds": state[
                    "time_seconds"
                ],
                "trigger_type": (
                    strongest_component
                ),
                "evidence": (
                    trigger_components
                ),
            }
        )

    return max(
        candidates,
        key=lambda candidate: candidate[
            "score"
        ],
    )


# ============================================================================
# RESPONSE ANALYSIS
# ============================================================================


def collect_episode_events(
    states: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    result = []

    seen = set()

    for state in states:

        for event in state.get(
            "motion_events",
            [],
        ):

            key = (
                event[
                    "entity_id"
                ],
                event[
                    "event"
                ],
                event[
                    "time_seconds"
                ],
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            result.append(
                event
            )

    result.sort(
        key=lambda event: event[
            "time_seconds"
        ]
    )

    return result


def calculate_state_response(
    state: Dict[str, Any],
) -> float:
    return clamp(
        safe_float(
            state.get(
                "acceleration",
                {},
            ).get(
                "normalized_magnitude",
                0.0,
            )
        )
    )


def calculate_track_disappearance_score(
    entity_history: Dict[
        str,
        List[Dict[str, Any]],
    ],
    entity_id: str,
    trigger_time: float,
    video_end_time: float,
) -> float:
    states = entity_history.get(
        entity_id,
        [],
    )

    if not states:
        return 0.0

    last_time = max(
        safe_float(
            state.get(
                "time_seconds"
            )
        )
        for state in states
    )

    after_trigger = (
        last_time
        >= trigger_time
    )

    disappears_soon = (
        0.0
        <= last_time
        -
        trigger_time
        <= TRACK_DISAPPEARANCE_WINDOW_SECONDS
    )

    not_video_end = (
        video_end_time
        -
        last_time
        >
        TRACK_DISAPPEARANCE_WINDOW_SECONDS
    )

    if (
        after_trigger
        and disappears_soon
        and not_video_end
    ):
        return 1.0

    return 0.0


def _local_pair_response_events(
    entity_history: Dict[str, List[Dict[str, Any]]],
    entity_ids: Tuple[str, str],
    trigger_time: float,
) -> List[Dict[str, Any]]:
    """Collect only events from the two interacting entities near the trigger."""
    start_time = trigger_time - LOCAL_EVENT_PRE_TRIGGER_SECONDS
    end_time = trigger_time + LOCAL_EVENT_POST_TRIGGER_SECONDS
    result = []
    seen = set()

    for entity_id in entity_ids:
        for state in entity_history.get(entity_id, []):
            state_time = safe_float(state.get("time_seconds"))
            if state_time < start_time:
                continue
            if state_time > end_time:
                break
            for event in state.get("motion_events", []):
                if not isinstance(event, dict):
                    continue
                event_time = safe_float(event.get("time_seconds", state_time))
                if not (start_time <= event_time <= end_time):
                    continue
                event_name = event.get("event", "unknown")
                key = (entity_id, event_name, round(event_time, 3))
                if key in seen:
                    continue
                seen.add(key)
                result.append({
                    "entity_id": entity_id,
                    "event": event_name,
                    "time_seconds": round(event_time, 3),
                    "confidence": round(clamp(safe_float(event.get("confidence"))), 4),
                })

    return sorted(result, key=lambda event: event["time_seconds"])


def _continuous_entity_response(
    entity_history: Dict[str, List[Dict[str, Any]]],
    entity_id: str,
    trigger_time: float,
) -> Dict[str, float]:
    """Measure continuous post-trigger motion change, not only discrete events."""
    history = entity_history.get(entity_id, [])
    baseline = [
        state for state in history
        if trigger_time - CONTINUOUS_BASELINE_SECONDS
        <= safe_float(state.get("time_seconds")) < trigger_time
    ]
    response = [
        state for state in history
        if trigger_time <= safe_float(state.get("time_seconds"))
        <= trigger_time + RESPONSE_WINDOW_SECONDS
    ]

    if not baseline or not response:
        return {
            "score": 0.0, "speed_change": 0.0,
            "direction_change": 0.0, "lateral_change": 0.0,
            "acceleration_change": 0.0,
        }

    def motion(state):
        value = state.get("relative_motion", {})
        return safe_float(value.get("vx")), safe_float(value.get("vy"))

    base_vx = float(np.median([motion(state)[0] for state in baseline]))
    base_vy = float(np.median([motion(state)[1] for state in baseline]))
    base_speed = math.hypot(base_vx, base_vy)
    base_angle = math.atan2(base_vy, base_vx) if base_speed > 1e-6 else None

    speed_change = 0.0
    direction_change = 0.0
    lateral_change = 0.0
    acceleration_change = 0.0

    for state in response:
        vx, vy = motion(state)
        speed = math.hypot(vx, vy)
        speed_change = max(speed_change, clamp(abs(speed - base_speed) / CONTINUOUS_SPEED_CHANGE_SCALE))
        lateral_change = max(lateral_change, clamp(abs(vx - base_vx) / CONTINUOUS_LATERAL_CHANGE_SCALE))

        if base_angle is not None and speed > 1e-6:
            angle = math.atan2(vy, vx)
            delta = abs(math.atan2(math.sin(angle - base_angle), math.cos(angle - base_angle)))
            direction_change = max(
                direction_change,
                clamp(math.degrees(delta) / CONTINUOUS_DIRECTION_CHANGE_DEGREES),
            )

        acceleration_change = max(acceleration_change, calculate_state_response(state))

    score = clamp(
        0.34 * speed_change
        + 0.28 * direction_change
        + 0.23 * lateral_change
        + 0.15 * acceleration_change
    )

    return {
        "score": score,
        "speed_change": speed_change,
        "direction_change": direction_change,
        "lateral_change": lateral_change,
        "acceleration_change": acceleration_change,
    }


def calculate_response_evidence(
    states: List[Dict[str, Any]],
    trigger_time: float,
    entity_history: Dict[str, List[Dict[str, Any]]],
    video_end_time: float,
) -> Dict[str, Any]:
    entity_a = states[0]["entity_a"]
    entity_b = states[0]["entity_b"]
    pair_entities = (entity_a, entity_b)

    # V5 does not inherit events from the whole episode. Events are fetched
    # directly from the two participants inside a short trigger-centred window.
    events = _local_pair_response_events(entity_history, pair_entities, trigger_time)
    response_events = []
    entity_event_scores = {entity_a: 0.0, entity_b: 0.0}
    strong_response_score = 0.0

    for event in events:
        delta_time = event["time_seconds"] - trigger_time
        if event["event"] not in RESPONSE_EVENTS:
            continue

        confidence = clamp(safe_float(event.get("confidence")))
        temporal_weight = clamp(1.0 - max(0.0, delta_time) / RESPONSE_WINDOW_SECONDS)
        event_score = clamp(confidence * (0.55 + 0.45 * temporal_weight))
        entity_id = event["entity_id"]
        entity_event_scores[entity_id] = max(entity_event_scores[entity_id], event_score)

        if event["event"] in STRONG_RESPONSE_EVENTS:
            strong_response_score = max(strong_response_score, event_score)

        response_events.append({
            **event,
            "delta_from_trigger": round(delta_time, 3),
            "response_score": round(event_score, 4),
        })

    continuous = {
        entity_a: _continuous_entity_response(entity_history, entity_a, trigger_time),
        entity_b: _continuous_entity_response(entity_history, entity_b, trigger_time),
    }

    disappearance = {
        entity_a: calculate_track_disappearance_score(entity_history, entity_a, trigger_time, video_end_time),
        entity_b: calculate_track_disappearance_score(entity_history, entity_b, trigger_time, video_end_time),
    }

    entity_scores = {}
    for entity_id in pair_entities:
        entity_scores[entity_id] = clamp(
            0.62 * entity_event_scores[entity_id]
            + 0.33 * continuous[entity_id]["score"]
            + 0.05 * disappearance[entity_id]
        )

    response_a = entity_scores[entity_a]
    response_b = entity_scores[entity_b]
    maximum_response = max(response_a, response_b)
    response_asymmetry = abs(response_a - response_b)

    return {
        "score": round(maximum_response, 4),
        "entity_a_response": round(response_a, 4),
        "entity_b_response": round(response_b, 4),
        "response_asymmetry": round(response_asymmetry, 4),
        "strong_response_score": round(strong_response_score, 4),
        "entity_a_continuous_response": {key: round(value, 4) for key, value in continuous[entity_a].items()},
        "entity_b_continuous_response": {key: round(value, 4) for key, value in continuous[entity_b].items()},
        "entity_a_acceleration_response": round(continuous[entity_a]["acceleration_change"], 4),
        "entity_b_acceleration_response": round(continuous[entity_b]["acceleration_change"], 4),
        "entity_a_disappearance": disappearance[entity_a] > 0.0,
        "entity_b_disappearance": disappearance[entity_b] > 0.0,
        "response_events": response_events,
        "response_attribution": "pair_specific_trigger_centered",
    }


# ============================================================================
# POST-TRIGGER SEPARATION
# ============================================================================


def calculate_post_trigger_separation(
    states: List[Dict[str, Any]],
    trigger_time: float,
) -> Dict[str, Any]:
    before_states = [
        state
        for state in states
        if (
            trigger_time
            -
            PRE_TRIGGER_WINDOW_SECONDS
            <= state[
                "time_seconds"
            ]
            <= trigger_time
        )
    ]

    after_states = [
        state
        for state in states
        if (
            trigger_time
            < state[
                "time_seconds"
            ]
            <= (
                trigger_time
                +
                RESPONSE_WINDOW_SECONDS
            )
        )
    ]

    if (
        not before_states
        or not after_states
    ):

        return {
            "score": 0.0,
            "pre_trigger_distance": None,
            "post_trigger_distance": None,
        }

    pre_distance = min(
        state[
            "geometry"
        ][
            "center_distance"
        ]
        for state in before_states
    )

    post_distance = max(
        state[
            "geometry"
        ][
            "center_distance"
        ]
        for state in after_states
    )

    separation_growth = clamp(
        (
            post_distance
            -
            pre_distance
        )
        /
        max(
            pre_distance,
            0.03,
        )
    )

    return {
        "score": round(
            separation_growth,
            4,
        ),
        "pre_trigger_distance": round(
            pre_distance,
            6,
        ),
        "post_trigger_distance": round(
            post_distance,
            6,
        ),
    }


# ============================================================================
# TRIGGER-RESPONSE CHAIN
# ============================================================================


def evaluate_event_chain(
    states: List[Dict[str, Any]],
    entity_history: Dict[
        str,
        List[Dict[str, Any]],
    ],
    video_end_time: float,
) -> Dict[str, Any]:
    trigger = (
        calculate_trigger_evidence(
            states
        )
    )

    trigger_time = trigger[
        "time_seconds"
    ]

    if trigger_time is None:

        return {
            "trigger": trigger,
            "response": {
                "score": 0.0,
            },
            "post_trigger_separation": {
                "score": 0.0,
            },
            "chain_score": 0.0,
            "trigger_present": False,
            "response_present": False,
            "temporal_chain_present": False,
            "critical_event_candidate": False,
        }

    response = (
        calculate_response_evidence(
            states,
            trigger_time,
            entity_history,
            video_end_time,
        )
    )

    separation = (
        calculate_post_trigger_separation(
            states,
            trigger_time,
        )
    )

    trigger_score = clamp(
        trigger[
            "score"
        ]
    )

    response_score = clamp(
        response[
            "score"
        ]
    )

    asymmetry_score = clamp(
        response[
            "response_asymmetry"
        ]
    )

    strong_response_score = clamp(
        response[
            "strong_response_score"
        ]
    )

    separation_score = clamp(
        separation[
            "score"
        ]
    )

    trigger_present = (
        trigger_score
        >= MIN_TRIGGER_SCORE
    )

    response_present = (
        response_score
        >= MIN_RESPONSE_SCORE
    )

    temporal_chain_present = (
        trigger_present
        and response_present
        and len(
            response[
                "response_events"
            ]
        )
        > 0
    )

    chain_score = clamp(
        0.36
        * trigger_score
        +
        0.34
        * response_score
        +
        0.12
        * strong_response_score
        +
        0.10
        * asymmetry_score
        +
        0.08
        * separation_score
    )

    if not temporal_chain_present:

        chain_score *= 0.45

    duration = (
        states[-1][
            "time_seconds"
        ]
        -
        states[0][
            "time_seconds"
        ]
    )

    short_event_bonus = (
        0.06
        if (
            duration
            <= SHORT_EVENT_SECONDS
            and temporal_chain_present
        )
        else 0.0
    )

    final_chain_score = clamp(
        chain_score
        +
        short_event_bonus
    )

    critical_event_candidate = (
        temporal_chain_present
        and final_chain_score
        >= CRITICAL_CHAIN_THRESHOLD
    )

    return {
        "trigger": {
            "score": round(
                trigger_score,
                4,
            ),
            "time_seconds": (
                trigger_time
            ),
            "trigger_type": (
                trigger[
                    "trigger_type"
                ]
            ),
            "evidence": (
                trigger[
                    "evidence"
                ]
            ),
        },
        "response": (
            response
        ),
        "post_trigger_separation": (
            separation
        ),
        "chain_score": round(
            final_chain_score,
            4,
        ),
        "trigger_present": (
            trigger_present
        ),
        "response_present": (
            response_present
        ),
        "temporal_chain_present": (
            temporal_chain_present
        ),
        "critical_event_candidate": (
            critical_event_candidate
        ),
    }


# ============================================================================
# INTERACTION EPISODE SUMMARY
# ============================================================================


def summarize_interaction_episode(
    states: List[Dict[str, Any]],
    entity_history: Dict[
        str,
        List[Dict[str, Any]],
    ],
    video_end_time: float,
) -> Dict[str, Any]:
    scores = [
        state[
            "interaction_score"
        ]
        for state in states
    ]

    peak_index = int(
        np.argmax(
            scores
        )
    )

    peak_state = states[
        peak_index
    ]

    interaction_types = [
        state[
            "interaction_type"
        ]
        for state in states
    ]

    dominant_type = max(
        set(
            interaction_types
        ),
        key=interaction_types.count,
    )

    event_chain = (
        evaluate_event_chain(
            states,
            entity_history,
            video_end_time,
        )
    )

    minimum_edge_distance = min(
        state[
            "geometry"
        ][
            "edge_distance"
        ]
        for state in states
    )

    maximum_overlap = max(
        state[
            "geometry"
        ][
            "bbox_overlap_ratio"
        ]
        for state in states
    )

    maximum_convergence = max(
        state[
            "trajectory"
        ][
            "convergence_score"
        ]
        for state in states
    )

    maximum_future_conflict = max(
        state[
            "trajectory"
        ][
            "future_conflict_score"
        ]
        for state in states
    )

    return {
        "entity_a": states[0][
            "entity_a"
        ],
        "entity_b": states[0][
            "entity_b"
        ],
        "visual_identity_a": states[0][
            "visual_identity_a"
        ],
        "visual_identity_b": states[0][
            "visual_identity_b"
        ],
        "class_a": states[0][
            "class_a"
        ],
        "class_b": states[0][
            "class_b"
        ],
        "start_time": states[0][
            "time_seconds"
        ],
        "end_time": states[-1][
            "time_seconds"
        ],
        "duration_seconds": round(
            states[-1][
                "time_seconds"
            ]
            -
            states[0][
                "time_seconds"
            ],
            3,
        ),
        "states": len(
            states
        ),
        "dominant_interaction_type": (
            dominant_type
        ),
        "interaction_types": sorted(
            set(
                interaction_types
            )
        ),
        "peak_interaction_score": round(
            scores[
                peak_index
            ],
            4,
        ),
        "peak_interaction_time": (
            peak_state[
                "time_seconds"
            ]
        ),
        "average_interaction_score": round(
            mean_or_zero(
                scores
            ),
            4,
        ),
        "minimum_edge_distance": round(
            minimum_edge_distance,
            6,
        ),
        "maximum_bbox_overlap_ratio": round(
            maximum_overlap,
            4,
        ),
        "maximum_convergence_score": round(
            maximum_convergence,
            4,
        ),
        "maximum_future_conflict_score": round(
            maximum_future_conflict,
            4,
        ),
        "vulnerable_road_user_pair": (
            states[0][
                "depth_compatibility"
            ][
                "vulnerable_road_user_pair"
            ]
        ),
        "event_chain": (
            event_chain
        ),
        "critical_event_candidate": (
            event_chain[
                "critical_event_candidate"
            ]
        ),
        "critical_event_score": (
            event_chain[
                "chain_score"
            ]
        ),
        "interpretation": (
            "Pairwise interaction episode. "
            "Overlap, proximity, and convergence are interaction "
            "context only. Critical-event eligibility requires a "
            "temporally linked trigger-response chain."
        ),
    }


# ============================================================================
# BUILD EPISODES
# ============================================================================


def build_interaction_episodes(
    timeline: List[Dict[str, Any]],
    entity_history: Dict[
        str,
        List[Dict[str, Any]],
    ],
    video_end_time: float,
) -> List[Dict[str, Any]]:
    pair_states = (
        collect_pair_states(
            timeline
        )
    )

    episodes = []

    for states in pair_states.values():

        pair_episodes = (
            split_into_episodes(
                states
            )
        )

        for episode_states in pair_episodes:

            episodes.append(
                summarize_interaction_episode(
                    episode_states,
                    entity_history,
                    video_end_time,
                )
            )

    episodes.sort(
        key=lambda episode: (
            episode[
                "critical_event_candidate"
            ],
            episode[
                "critical_event_score"
            ],
            episode[
                "event_chain"
            ][
                "response"
            ].get(
                "strong_response_score",
                0.0,
            ),
            episode[
                "event_chain"
            ][
                "response"
            ].get(
                "response_asymmetry",
                0.0,
            ),
            episode[
                "peak_interaction_score"
            ],
        ),
        reverse=True,
    )

    for index, episode in enumerate(
        episodes,
        start=1,
    ):

        episode[
            "interaction_episode_id"
        ] = (
            f"Interaction_{index}"
        )

    return episodes


# ============================================================================
# CRITICAL EVENT CANDIDATES
# ============================================================================


def build_critical_event_candidates(
    episodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates = []

    for episode in episodes:

        if not episode[
            "critical_event_candidate"
        ]:
            continue

        event_chain = episode[
            "event_chain"
        ]

        candidates.append(
            {
                "entity_a": (
                    episode[
                        "entity_a"
                    ]
                ),
                "entity_b": (
                    episode[
                        "entity_b"
                    ]
                ),
                "visual_identity_a": (
                    episode[
                        "visual_identity_a"
                    ]
                ),
                "visual_identity_b": (
                    episode[
                        "visual_identity_b"
                    ]
                ),
                "class_a": (
                    episode[
                        "class_a"
                    ]
                ),
                "class_b": (
                    episode[
                        "class_b"
                    ]
                ),
                "interaction_episode_id": (
                    episode[
                        "interaction_episode_id"
                    ]
                ),
                "event_start_time": (
                    episode[
                        "start_time"
                    ]
                ),
                "trigger_time": (
                    event_chain[
                        "trigger"
                    ][
                        "time_seconds"
                    ]
                ),
                "event_end_time": (
                    episode[
                        "end_time"
                    ]
                ),
                "trigger_type": (
                    event_chain[
                        "trigger"
                    ][
                        "trigger_type"
                    ]
                ),
                "trigger_score": (
                    event_chain[
                        "trigger"
                    ][
                        "score"
                    ]
                ),
                "response_score": (
                    event_chain[
                        "response"
                    ][
                        "score"
                    ]
                ),
                "response_asymmetry": (
                    event_chain[
                        "response"
                    ][
                        "response_asymmetry"
                    ]
                ),
                "post_trigger_separation_score": (
                    event_chain[
                        "post_trigger_separation"
                    ][
                        "score"
                    ]
                ),
                "critical_event_score": (
                    episode[
                        "critical_event_score"
                    ]
                ),
                "supporting_response_events": (
                    event_chain[
                        "response"
                    ][
                        "response_events"
                    ]
                ),
                "evidence_statement": (
                    "Critical-event candidate because a pairwise "
                    "interaction trigger is temporally followed by "
                    "exceptional participant motion response. "
                    "This is not collision confirmation."
                ),
            }
        )

    candidates.sort(
        key=lambda candidate: candidate[
            "critical_event_score"
        ],
        reverse=True,
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        candidate[
            "critical_event_id"
        ] = (
            f"CriticalEvent_{index}"
        )

    return candidates


# ============================================================================
# ANALYSIS
# ============================================================================


def analyze_interactions(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    entity_history = (
        build_entity_history(
            data
        )
    )

    all_times = [
        safe_float(
            state.get(
                "time_seconds"
            )
        )
        for states in entity_history.values()
        for state in states
    ]

    video_end_time = (
        max(
            all_times
        )
        if all_times
        else 0.0
    )

    timeline = (
        build_interaction_timeline(
            data
        )
    )

    episodes = (
        build_interaction_episodes(
            timeline,
            entity_history,
            video_end_time,
        )
    )

    critical_events = (
        build_critical_event_candidates(
            episodes
        )
    )

    return {
        "configuration": {
            "version": "v5",
            "standalone": True,
            "causal_pairwise_analysis": True,
            "event_chain_analysis": True,
            "pair_specific_response_attribution": True,
            "continuous_response_evidence": True,
            "trigger_centered_event_window": True,
            "trigger_response_required": True,
            "bbox_overlap_is_collision": False,
            "bbox_overlap_direct_critical_weight": 0.0,
            "proximity_direct_critical_weight": 0.0,
            "edge_distance_direct_critical_weight": 0.0,
            "convergence_alone_is_critical": False,
            "future_conflict_alone_is_critical": False,
            "critical_event_requires_temporal_response": True,
            "accident_classification": False,
            "pre_accident_alerting": False,
            "design_note": (
                "V5 preserves the V4 geometry/trigger separation and adds pair-specific, trigger-centered response attribution plus continuous motion-change evidence "
                "from critical-event evidence. Geometry can establish "
                "a trigger context but cannot directly create a critical "
                "event. Critical-event candidates require a temporally "
                "linked trigger-response chain."
            ),
        },
        "interaction_timeline": (
            timeline
        ),
        "interaction_episodes": (
            episodes
        ),
        "critical_event_candidates": (
            critical_events
        ),
        "summary": {
            "timeline_states": len(
                timeline
            ),
            "interaction_episodes": len(
                episodes
            ),
            "pairs_with_episodes": len(
                {
                    pair_key(
                        episode[
                            "entity_a"
                        ],
                        episode[
                            "entity_b"
                        ],
                    )
                    for episode
                    in episodes
                }
            ),
            "critical_event_candidates": len(
                critical_events
            ),
        },
    }


# ============================================================================
# OUTPUT
# ============================================================================


def derive_output_path(
    input_path: Path,
) -> Path:
    stem = input_path.stem

    if stem.endswith(
        "_object_analysis"
    ):

        stem = stem[
            :-len(
                "_object_analysis"
            )
        ]

    return input_path.with_name(
        f"{stem}_interaction_analysis.json"
    )


def print_summary(
    result: Dict[str, Any],
) -> None:
    print(
        "\n"
        +
        "=" * 88
    )

    print(
        "INTERACTION ANALYZER V5 COMPLETE"
    )

    print(
        "=" * 88
    )

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

    candidates = result[
        "critical_event_candidates"
    ]

    if not candidates:

        print(
            "\nNo trigger-response critical "
            "event chain was confirmed."
        )

    else:

        print(
            "\nCritical event candidates:"
        )

        for candidate in candidates[:20]:

            print(
                f"\n"
                f"{candidate['critical_event_id']}  "
                f"{candidate['visual_identity_a']} "
                f"<-> "
                f"{candidate['visual_identity_b']}"
            )

            print(
                f"  Classes       : "
                f"{candidate['class_a']} "
                f"<-> "
                f"{candidate['class_b']}"
            )

            print(
                f"  Event window  : "
                f"{candidate['event_start_time']:.2f} - "
                f"{candidate['event_end_time']:.2f} s"
            )

            print(
                f"  Trigger time  : "
                f"{candidate['trigger_time']:.2f} s"
            )

            print(
                f"  Trigger type  : "
                f"{candidate['trigger_type']}"
            )

            print(
                f"  Trigger score : "
                f"{candidate['trigger_score']:.4f}"
            )

            print(
                f"  Response      : "
                f"{candidate['response_score']:.4f}"
            )

            print(
                f"  Asymmetry     : "
                f"{candidate['response_asymmetry']:.4f}"
            )

            print(
                f"  Chain score   : "
                f"{candidate['critical_event_score']:.4f}"
            )

    print(
        "\n"
        +
        "=" * 88
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone V5 trigger-response "
            "traffic interaction analyzer."
        )
    )

    parser.add_argument(
        "object_analysis_json",
        type=Path,
        help=(
            "Path to *_object_analysis.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output path. Default: "
            "<video>_interaction_analysis.json"
        ),
    )

    args = parser.parse_args()

    data = load_json(
        args.object_analysis_json
    )

    result = analyze_interactions(
        data
    )

    output_path = (
        args.output
        if args.output is not None
        else derive_output_path(
            args.object_analysis_json
        )
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
        f"\nSaved V5 interaction analysis to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
