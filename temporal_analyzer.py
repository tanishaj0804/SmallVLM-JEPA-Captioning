import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5

TOP_K = 5

MIN_ENTITY_STATES = 3
MIN_PAIR_SHARED_STATES = 3

EPS = 1e-8


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def euclidean(point_a, point_b):
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2
        + (point_a[1] - point_b[1]) ** 2
    )


def robust_normalize(values):
    """
    Robust percentile normalization into [0, 1].

    No event-specific threshold is used.
    """

    if not values:
        return []

    array = np.asarray(values, dtype=np.float64)

    low = float(np.percentile(array, 10))
    high = float(np.percentile(array, 90))

    if abs(high - low) < EPS:
        return [0.0 for _ in values]

    normalized = (
        (array - low)
        / (high - low)
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return normalized.tolist()


def median_or_zero(values):
    if not values:
        return 0.0

    return float(np.median(values))


def max_or_zero(values):
    if not values:
        return 0.0

    return float(np.max(values))


# ============================================================
# INPUT VALIDATION
# ============================================================

def load_entity_states(data):
    """
    Exact object_analyzer.py output contract.

    Expected:

    {
        "configuration": {...},
        "scene_flow_timeline": [...],
        "entity_states": {
            "Car_1": [...],
            "Motorcycle_1": [...],
            ...
        },
        "entity_summaries": [...]
    }
    """

    entity_states = data.get("entity_states")

    if not isinstance(entity_states, dict):
        raise ValueError(
            "Expected 'entity_states' dictionary "
            "in object analysis JSON."
        )

    cleaned = {}

    for entity_id, states in entity_states.items():

        if not isinstance(states, list):
            continue

        valid_states = []

        for state in states:

            if not isinstance(state, dict):
                continue

            if "time_seconds" not in state:
                continue

            geometry = state.get("geometry")

            if not isinstance(geometry, dict):
                continue

            if (
                "center_x" not in geometry
                or "center_y" not in geometry
            ):
                continue

            valid_states.append(state)

        valid_states.sort(
            key=lambda state: safe_float(
                state.get("time_seconds")
            )
        )

        if valid_states:
            cleaned[entity_id] = valid_states

    if not cleaned:
        raise ValueError(
            "No valid entity timelines found."
        )

    return cleaned


# ============================================================
# TEMPORAL HELPERS
# ============================================================

def get_states_in_window(
    states,
    start_time,
    end_time,
):
    return [
        state
        for state in states
        if (
            start_time
            <= safe_float(state["time_seconds"])
            < end_time
        )
    ]


def get_state_center(state):
    geometry = state["geometry"]

    return (
        safe_float(geometry.get("center_x")),
        safe_float(geometry.get("center_y")),
    )


def get_state_area(state):
    return safe_float(
        state.get(
            "geometry",
            {},
        ).get("area")
    )


def get_state_width(state):
    return safe_float(
        state.get(
            "geometry",
            {},
        ).get("width")
    )


def get_state_height(state):
    return safe_float(
        state.get(
            "geometry",
            {},
        ).get("height")
    )


def get_relative_speed(state):
    return safe_float(
        state.get(
            "relative_motion",
            {},
        ).get("speed")
    )


def get_relative_velocity(state):
    relative_motion = state.get(
        "relative_motion",
        {},
    )

    return (
        safe_float(relative_motion.get("vx")),
        safe_float(relative_motion.get("vy")),
    )


def get_acceleration(state):
    return safe_float(
        state.get(
            "acceleration",
            {},
        ).get("magnitude")
    )


def get_motion_confidence(state):
    return safe_float(
        state.get(
            "motion_quality",
            {},
        ).get("motion_confidence")
    )


# ============================================================
# WINDOW GENERATION
# ============================================================

def generate_windows(entity_states):
    timestamps = []

    for states in entity_states.values():
        timestamps.extend(
            safe_float(state["time_seconds"])
            for state in states
        )

    if not timestamps:
        return []

    video_start = min(timestamps)
    video_end = max(timestamps)

    windows = []

    current = video_start
    window_id = 0

    while current <= video_end:

        windows.append(
            {
                "window_id": window_id,
                "start_time": current,
                "end_time": current + WINDOW_SECONDS,
            }
        )

        current += STRIDE_SECONDS
        window_id += 1

    return windows


# ============================================================
# SIGNAL 1
# MOTION CHANGE
# ============================================================

def calculate_motion_change(
    entity_states,
    start_time,
    end_time,
):
    """
    Measures sudden single-entity motion variation.

    Uses object_analyzer outputs directly:

        relative speed
        acceleration
        velocity direction change
    """

    entity_scores = {}

    for entity_id, timeline in entity_states.items():

        states = get_states_in_window(
            timeline,
            start_time,
            end_time,
        )

        if len(states) < MIN_ENTITY_STATES:
            continue

        speeds = [
            get_relative_speed(state)
            for state in states
        ]

        accelerations = [
            get_acceleration(state)
            for state in states
        ]

        velocity_vectors = [
            get_relative_velocity(state)
            for state in states
        ]

        speed_changes = []

        for index in range(1, len(speeds)):
            speed_changes.append(
                abs(
                    speeds[index]
                    - speeds[index - 1]
                )
            )

        direction_changes = []

        for index in range(
            1,
            len(velocity_vectors),
        ):

            vx_previous, vy_previous = (
                velocity_vectors[index - 1]
            )

            vx_current, vy_current = (
                velocity_vectors[index]
            )

            magnitude_previous = math.sqrt(
                vx_previous ** 2
                + vy_previous ** 2
            )

            magnitude_current = math.sqrt(
                vx_current ** 2
                + vy_current ** 2
            )

            if (
                magnitude_previous < EPS
                or magnitude_current < EPS
            ):
                continue

            dot_product = (
                vx_previous * vx_current
                + vy_previous * vy_current
            )

            cosine = dot_product / (
                magnitude_previous
                * magnitude_current
            )

            cosine = np.clip(
                cosine,
                -1.0,
                1.0,
            )

            angle = math.acos(cosine)

            direction_changes.append(angle)

        speed_component = max_or_zero(
            speed_changes
        )

        acceleration_component = max_or_zero(
            accelerations
        )

        direction_component = max_or_zero(
            direction_changes
        )

        confidence = median_or_zero(
            [
                get_motion_confidence(state)
                for state in states
            ]
        )

        raw_score = (
            speed_component
            + acceleration_component
            + direction_component
        )

        score = raw_score * (
            0.5 + 0.5 * confidence
        )

        entity_scores[entity_id] = score

    if not entity_scores:
        return 0.0, []

    ranked = sorted(
        entity_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_scores = [
        score
        for _, score in ranked[:3]
    ]

    return (
        float(np.mean(top_scores)),
        [
            entity_id
            for entity_id, _ in ranked[:3]
        ],
    )


# ============================================================
# SIGNAL 2
# RAPID PROXIMITY CHANGE
# ============================================================

def calculate_proximity_change(
    entity_states,
    start_time,
    end_time,
):
    """
    Measures rapid pairwise approach.

    This is NOT collision detection.

    It only asks:

        Did two visible entities rapidly become closer?
    """

    active_entities = {}

    for entity_id, timeline in entity_states.items():

        states = get_states_in_window(
            timeline,
            start_time,
            end_time,
        )

        if len(states) >= MIN_PAIR_SHARED_STATES:
            active_entities[entity_id] = states

    entity_ids = list(active_entities.keys())

    pair_scores = {}

    for index_a in range(len(entity_ids)):

        for index_b in range(
            index_a + 1,
            len(entity_ids),
        ):

            entity_a = entity_ids[index_a]
            entity_b = entity_ids[index_b]

            states_a = active_entities[entity_a]
            states_b = active_entities[entity_b]

            frame_map_a = {
                state["frame"]: state
                for state in states_a
            }

            frame_map_b = {
                state["frame"]: state
                for state in states_b
            }

            shared_frames = sorted(
                set(frame_map_a)
                & set(frame_map_b)
            )

            if (
                len(shared_frames)
                < MIN_PAIR_SHARED_STATES
            ):
                continue

            distances = []

            timestamps = []

            for frame in shared_frames:

                state_a = frame_map_a[frame]
                state_b = frame_map_b[frame]

                center_a = get_state_center(
                    state_a
                )

                center_b = get_state_center(
                    state_b
                )

                distances.append(
                    euclidean(
                        center_a,
                        center_b,
                    )
                )

                timestamps.append(
                    safe_float(
                        state_a["time_seconds"]
                    )
                )

            closing_rates = []

            for index in range(
                1,
                len(distances),
            ):

                dt = (
                    timestamps[index]
                    - timestamps[index - 1]
                )

                if dt <= EPS:
                    continue

                distance_change = (
                    distances[index - 1]
                    - distances[index]
                )

                closing_rate = (
                    distance_change / dt
                )

                if closing_rate > 0:
                    closing_rates.append(
                        closing_rate
                    )

            if not closing_rates:
                continue

            peak_closing_rate = max(
                closing_rates
            )

            total_approach = max(
                0.0,
                distances[0]
                - min(distances),
            )

            minimum_distance = min(distances)

            proximity_weight = 1.0 / (
                minimum_distance + 0.05
            )

            score = (
                peak_closing_rate
                + total_approach
            ) * proximity_weight

            pair_name = (
                f"{entity_a}|{entity_b}"
            )

            pair_scores[pair_name] = score

    if not pair_scores:
        return 0.0, []

    ranked = sorted(
        pair_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_scores = [
        score
        for _, score in ranked[:3]
    ]

    return (
        float(np.mean(top_scores)),
        [
            pair_name
            for pair_name, _ in ranked[:3]
        ],
    )


# ============================================================
# SIGNAL 3
# TRACK DISRUPTION
# ============================================================

def calculate_track_disruption(
    entity_states,
    start_time,
    end_time,
):
    """
    Detects continuity breakdown.

    Examples:

        stable track disappears
        observation gaps increase
        entity becomes unstable near window end

    Track loss is treated as anomaly evidence,
    NOT accident evidence.
    """

    entity_scores = {}

    for entity_id, timeline in entity_states.items():

        before_states = get_states_in_window(
            timeline,
            start_time - WINDOW_SECONDS,
            start_time,
        )

        inside_states = get_states_in_window(
            timeline,
            start_time,
            end_time,
        )

        after_states = get_states_in_window(
            timeline,
            end_time,
            end_time + WINDOW_SECONDS,
        )

        if len(inside_states) < MIN_ENTITY_STATES:
            continue

        inside_times = [
            safe_float(state["time_seconds"])
            for state in inside_states
        ]

        gaps = []

        for index in range(
            1,
            len(inside_times),
        ):

            gaps.append(
                inside_times[index]
                - inside_times[index - 1]
            )

        gap_score = 0.0

        if gaps:

            median_gap = np.median(gaps)

            maximum_gap = max(gaps)

            gap_score = max(
                0.0,
                maximum_gap
                - median_gap
            )

        history_score = 0.0

        if (
            len(before_states)
            >= MIN_ENTITY_STATES
        ):

            before_density = (
                len(before_states)
                / WINDOW_SECONDS
            )

            after_density = (
                len(after_states)
                / WINDOW_SECONDS
            )

            if before_density > EPS:

                history_score = max(
                    0.0,
                    (
                        before_density
                        - after_density
                    )
                    / before_density,
                )

        last_state_time = safe_float(
            inside_states[-1]["time_seconds"]
        )

        early_termination = max(
            0.0,
            end_time - last_state_time,
        )

        score = (
            gap_score
            + history_score
            + early_termination
        )

        entity_scores[entity_id] = score

    if not entity_scores:
        return 0.0, []

    ranked = sorted(
        entity_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_scores = [
        score
        for _, score in ranked[:3]
    ]

    return (
        float(np.mean(top_scores)),
        [
            entity_id
            for entity_id, _ in ranked[:3]
        ],
    )


# ============================================================
# SIGNAL 4
# GEOMETRY CHANGE
# ============================================================

def calculate_geometry_change(
    entity_states,
    start_time,
    end_time,
):
    """
    Measures unusual bounding-box deformation.

    Useful for:

        motorcycle fall
        person fall
        vehicle rotation
        partial occlusion
        sudden pose change

    Geometry change is not classified here.
    """

    entity_scores = {}

    for entity_id, timeline in entity_states.items():

        states = get_states_in_window(
            timeline,
            start_time,
            end_time,
        )

        if len(states) < MIN_ENTITY_STATES:
            continue

        areas = [
            get_state_area(state)
            for state in states
            if get_state_area(state) > EPS
        ]

        widths = [
            get_state_width(state)
            for state in states
            if get_state_width(state) > EPS
        ]

        heights = [
            get_state_height(state)
            for state in states
            if get_state_height(state) > EPS
        ]

        if len(areas) < 2:
            continue

        area_changes = []

        for index in range(1, len(areas)):

            previous = areas[index - 1]

            current = areas[index]

            area_changes.append(
                abs(current - previous)
                / max(previous, EPS)
            )

        aspect_ratios = []

        for width, height in zip(
            widths,
            heights,
        ):

            aspect_ratios.append(
                width / max(height, EPS)
            )

        aspect_changes = []

        for index in range(
            1,
            len(aspect_ratios),
        ):

            previous = aspect_ratios[
                index - 1
            ]

            current = aspect_ratios[index]

            aspect_changes.append(
                abs(current - previous)
                / max(previous, EPS)
            )

        area_peak = max_or_zero(
            area_changes
        )

        aspect_peak = max_or_zero(
            aspect_changes
        )

        area_variability = float(
            np.std(areas)
            / max(
                np.mean(areas),
                EPS,
            )
        )

        score = (
            area_peak
            + aspect_peak
            + area_variability
        )

        entity_scores[entity_id] = score

    if not entity_scores:
        return 0.0, []

    ranked = sorted(
        entity_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_scores = [
        score
        for _, score in ranked[:3]
    ]

    return (
        float(np.mean(top_scores)),
        [
            entity_id
            for entity_id, _ in ranked[:3]
        ],
    )


# ============================================================
# SIGNAL 5
# LOCAL MOTION CHAOS
# ============================================================

def calculate_local_motion_chaos(
    entity_states,
    start_time,
    end_time,
):
    """
    Measures simultaneous motion instability.

    A critical event may affect several nearby tracks:

        braking
        swerving
        falling
        occlusion
        tracker instability

    This signal looks for collective motion disorder.
    """

    entity_variabilities = {}

    for entity_id, timeline in entity_states.items():

        states = get_states_in_window(
            timeline,
            start_time,
            end_time,
        )

        if len(states) < MIN_ENTITY_STATES:
            continue

        speeds = np.asarray(
            [
                get_relative_speed(state)
                for state in states
            ],
            dtype=np.float64,
        )

        accelerations = np.asarray(
            [
                get_acceleration(state)
                for state in states
            ],
            dtype=np.float64,
        )

        vx_values = np.asarray(
            [
                get_relative_velocity(state)[0]
                for state in states
            ],
            dtype=np.float64,
        )

        vy_values = np.asarray(
            [
                get_relative_velocity(state)[1]
                for state in states
            ],
            dtype=np.float64,
        )

        variability = (
            np.std(speeds)
            + np.std(accelerations)
            + np.std(vx_values)
            + np.std(vy_values)
        )

        confidence = median_or_zero(
            [
                get_motion_confidence(state)
                for state in states
            ]
        )

        variability *= (
            0.5 + 0.5 * confidence
        )

        entity_variabilities[
            entity_id
        ] = float(variability)

    if len(entity_variabilities) < 2:
        return 0.0, []

    ranked = sorted(
        entity_variabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_entities = ranked[:5]

    values = [
        score
        for _, score in top_entities
    ]

    chaos_score = (
        float(np.mean(values))
        * math.log1p(len(values))
    )

    return (
        chaos_score,
        [
            entity_id
            for entity_id, _ in top_entities
        ],
    )


# ============================================================
# SIGNAL 6
# MOTION EVENT DENSITY
# ============================================================

def calculate_motion_event_density(
    entity_states,
    start_time,
    end_time,
):
    """
    Uses transition events already generated by object_analyzer.

    We do NOT interpret the event semantically.

    We only measure how many motion transitions occur
    in the temporal window.
    """

    entity_scores = {}

    for entity_id, timeline in entity_states.items():

        events = []

        seen_events = set()

        for state in timeline:

            state_time = safe_float(
                state["time_seconds"]
            )

            if not (
                start_time
                <= state_time
                < end_time
            ):
                continue

            for event in state.get(
                "motion_events",
                [],
            ):

                event_time = safe_float(
                    event.get(
                        "time_seconds",
                        state_time,
                    )
                )

                event_name = str(
                    event.get(
                        "event",
                        "unknown",
                    )
                )

                event_key = (
                    round(event_time, 4),
                    event_name,
                )

                if event_key in seen_events:
                    continue

                seen_events.add(event_key)

                confidence = safe_float(
                    event.get(
                        "confidence",
                        0.0,
                    )
                )

                events.append(
                    (
                        event_time,
                        event_name,
                        confidence,
                    )
                )

        if not events:
            continue

        confidence_sum = sum(
            confidence
            for _, _, confidence in events
        )

        temporal_density = (
            len(events)
            / WINDOW_SECONDS
        )

        score = (
            temporal_density
            * (
                0.5
                + confidence_sum
                / max(
                    2.0 * len(events),
                    EPS,
                )
            )
        )

        entity_scores[entity_id] = score

    if not entity_scores:
        return 0.0, []

    ranked = sorted(
        entity_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_scores = [
        score
        for _, score in ranked[:3]
    ]

    return (
        float(np.mean(top_scores)),
        [
            entity_id
            for entity_id, _ in ranked[:3]
        ],
    )


# ============================================================
# WINDOW ANALYSIS
# ============================================================

def analyze_windows(
    windows,
    entity_states,
):
    results = []

    for window in windows:

        start_time = window["start_time"]
        end_time = window["end_time"]

        (
            motion_score,
            motion_entities,
        ) = calculate_motion_change(
            entity_states,
            start_time,
            end_time,
        )

        (
            proximity_score,
            proximity_pairs,
        ) = calculate_proximity_change(
            entity_states,
            start_time,
            end_time,
        )

        (
            disruption_score,
            disruption_entities,
        ) = calculate_track_disruption(
            entity_states,
            start_time,
            end_time,
        )

        (
            geometry_score,
            geometry_entities,
        ) = calculate_geometry_change(
            entity_states,
            start_time,
            end_time,
        )

        (
            chaos_score,
            chaos_entities,
        ) = calculate_local_motion_chaos(
            entity_states,
            start_time,
            end_time,
        )

        (
            event_density_score,
            event_entities,
        ) = calculate_motion_event_density(
            entity_states,
            start_time,
            end_time,
        )

        results.append(
            {
                "window_id": window["window_id"],
                "start_time": round(
                    start_time,
                    3,
                ),
                "end_time": round(
                    end_time,
                    3,
                ),

                "raw_signals": {
                    "motion_change": motion_score,
                    "proximity_change": proximity_score,
                    "track_disruption": disruption_score,
                    "geometry_change": geometry_score,
                    "local_motion_chaos": chaos_score,
                    "motion_event_density": event_density_score,
                },

                "evidence": {
                    "motion_entities": motion_entities,
                    "proximity_pairs": proximity_pairs,
                    "disruption_entities": disruption_entities,
                    "geometry_entities": geometry_entities,
                    "chaos_entities": chaos_entities,
                    "motion_event_entities": event_entities,
                },
            }
        )

    return results


# ============================================================
# NORMALIZATION AND RANKING
# ============================================================
def normalize_and_rank(window_results):
    signal_names = [
        "motion_change",
        "proximity_change",
        "track_disruption",
        "geometry_change",
        "local_motion_chaos",
        "motion_event_density",
    ]

    normalized_signals = {}

    # ========================================================
    # NORMALIZE EACH SIGNAL ACROSS THE VIDEO
    # ========================================================

    for signal_name in signal_names:

        values = [
            result["raw_signals"][signal_name]
            for result in window_results
        ]

        normalized_signals[signal_name] = (
            robust_normalize(values)
        )

    # ========================================================
    # ATTACH NORMALIZED SIGNALS
    # ========================================================

    for index, result in enumerate(window_results):

        normalized = {}

        for signal_name in signal_names:

            normalized[signal_name] = round(
                normalized_signals[
                    signal_name
                ][index],
                4,
            )

        result["normalized_signals"] = normalized

    # ========================================================
    # TEMPORAL TRANSITION ANALYSIS
    # ========================================================

    transition_raw_scores = []

    for index, result in enumerate(window_results):

        current_vector = np.asarray(
            [
                result["normalized_signals"][
                    signal_name
                ]
                for signal_name in signal_names
            ],
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # No causal history exists for first window
        # ----------------------------------------------------

        if index == 0:

            transition_raw_scores.append(0.0)

            result["transition_evidence"] = {
                "positive_signal_changes": {},
                "history_windows_used": 0,
            }

            continue

        # ----------------------------------------------------
        # Use up to two previous windows as causal baseline
        # ----------------------------------------------------

        history_start = max(
            0,
            index - 2,
        )

        history_vectors = []

        for history_index in range(
            history_start,
            index,
        ):

            history_result = window_results[
                history_index
            ]

            history_vector = np.asarray(
                [
                    history_result[
                        "normalized_signals"
                    ][signal_name]
                    for signal_name in signal_names
                ],
                dtype=np.float64,
            )

            history_vectors.append(
                history_vector
            )

        history_baseline = np.median(
            np.asarray(history_vectors),
            axis=0,
        )

        signal_changes = (
            current_vector
            - history_baseline
        )

        # ----------------------------------------------------
        # We care about abnormality ONSET.
        #
        # Negative change means behaviour became calmer.
        # That is not event onset evidence.
        # ----------------------------------------------------

        positive_changes = np.maximum(
            signal_changes,
            0.0,
        )

        active_changes = positive_changes[
            positive_changes > 0.0
        ]

        if len(active_changes) == 0:

            transition_score = 0.0

        else:

            mean_positive_change = float(
                np.mean(active_changes)
            )

            peak_positive_change = float(
                np.max(active_changes)
            )

            changed_signal_ratio = (
                len(active_changes)
                / len(signal_names)
            )

            transition_score = (
                0.45 * mean_positive_change
                + 0.35 * peak_positive_change
                + 0.20 * changed_signal_ratio
            )

        transition_raw_scores.append(
            transition_score
        )

        positive_signal_changes = {}

        for signal_index, signal_name in enumerate(
            signal_names
        ):

            change = positive_changes[
                signal_index
            ]

            if change > 0.0:

                positive_signal_changes[
                    signal_name
                ] = round(
                    float(change),
                    4,
                )

        result["transition_evidence"] = {
            "positive_signal_changes": (
                positive_signal_changes
            ),
            "history_windows_used": len(
                history_vectors
            ),
        }

    # ========================================================
    # NORMALIZE TRANSITION SCORES
    # ========================================================

    normalized_transition_scores = (
        robust_normalize(
            transition_raw_scores
        )
    )

    # ========================================================
    # FINAL TEMPORAL ABNORMALITY SCORE
    # ========================================================

    for index, result in enumerate(
        window_results
    ):

        signal_values = list(
            result[
                "normalized_signals"
            ].values()
        )

        state_mean = float(
            np.mean(signal_values)
        )

        state_peak = float(
            np.max(signal_values)
        )

        state_score = (
            0.75 * state_mean
            + 0.25 * state_peak
        )

        transition_score = (
            normalized_transition_scores[
                index
            ]
        )

        # ----------------------------------------------------
        # Final score:
        #
        # 55% current abnormal state
        # 45% temporal surprise
        #
        # A critical candidate should ideally be both:
        #
        #   abnormal
        #       AND
        #   unexpectedly different from recent history
        # ----------------------------------------------------

        abnormality_score = (
            0.55 * state_score
            + 0.45 * transition_score
        )

        result["state_score"] = round(
            state_score,
            4,
        )

        result["transition_score"] = round(
            float(transition_score),
            4,
        )

        result["abnormality_score"] = round(
            abnormality_score,
            4,
        )

    # ========================================================
    # RANK
    # ========================================================

    ranked = sorted(
        window_results,
        key=lambda result: result[
            "abnormality_score"
        ],
        reverse=True,
    )

    for rank, result in enumerate(
        ranked,
        start=1,
    ):

        result["rank"] = rank

    return ranked


# ============================================================
# SUMMARY
# ============================================================

def print_summary(ranked_results):
    print()

    print("=" * 100)

    print(
        "TEMPORAL ABNORMALITY ANALYZER "
        "BASELINE COMPLETE"
    )

    print("=" * 100)

    print()

    print(
        f"Temporal windows analyzed : "
        f"{len(ranked_results)}"
    )

    print(
        f"Top candidate windows     : "
        f"{min(TOP_K, len(ranked_results))}"
    )

    print()

    print("-" * 100)

    for result in ranked_results[:TOP_K]:

        signals = result[
            "normalized_signals"
        ]

        print(
            f"RANK {result['rank']:02d} | "
            f"{result['start_time']:.2f}s"
            f" -> "
            f"{result['end_time']:.2f}s"
            f" | SCORE "
            f"{result['abnormality_score']:.4f}"
        )

        print(
            f"  motion change        : "
            f"{signals['motion_change']:.4f}"
        )

        print(
            f"  proximity change     : "
            f"{signals['proximity_change']:.4f}"
        )

        print(
            f"  track disruption     : "
            f"{signals['track_disruption']:.4f}"
        )

        print(
            f"  geometry change      : "
            f"{signals['geometry_change']:.4f}"
        )

        print(
            f"  local motion chaos   : "
            f"{signals['local_motion_chaos']:.4f}"
        )

        print(
            f"  motion event density : "
            f"{signals['motion_event_density']:.4f}"
        )

        print(
            "  evidence:"
        )

        evidence = result["evidence"]

        print(
            f"    motion      -> "
            f"{evidence['motion_entities']}"
        )

        print(
            f"    proximity   -> "
            f"{evidence['proximity_pairs']}"
        )

        print(
            f"    disruption  -> "
            f"{evidence['disruption_entities']}"
        )

        print(
            f"    geometry    -> "
            f"{evidence['geometry_entities']}"
        )

        print(
            f"    chaos       -> "
            f"{evidence['chaos_entities']}"
        )

        print(
            f"    events      -> "
            f"{evidence['motion_event_entities']}"
        )

        print("-" * 100)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_json",
        help=(
            "Path to object_analyzer output JSON"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path",
    )

    args = parser.parse_args()

    input_path = Path(args.input_json)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input JSON not found: "
            f"{input_path}"
        )

    print(
        f"[temporal] Loading object analysis: "
        f"{input_path}"
    )

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    entity_states = load_entity_states(data)

    total_states = sum(
        len(states)
        for states in entity_states.values()
    )

    print(
        f"[temporal] Entities discovered: "
        f"{len(entity_states)}"
    )

    print(
        f"[temporal] Entity states: "
        f"{total_states}"
    )

    windows = generate_windows(
        entity_states
    )

    print(
        f"[temporal] Temporal windows: "
        f"{len(windows)}"
    )



    window_results = analyze_windows(
        windows,
        entity_states,
    )

    ranked_results = normalize_and_rank(
        window_results
    )

    if args.output is None:

        output_name = (
            input_path.stem.replace(
                "_object_analysis",
                "",
            )
            + "_temporal_analysis.json"
        )

        output_path = (
            input_path.parent
            / output_name
        )

    else:

        output_path = Path(args.output)

    output_data = {
        "configuration": {
            "window_seconds": WINDOW_SECONDS,
            "stride_seconds": STRIDE_SECONDS,
            "top_k": TOP_K,
            "ranking_method": (
                "robust normalized multi-signal "
                "temporal abnormality"
            ),
            "accident_classification": False,
            "interaction_classification": False,
            "design_note": (
                "This stage ranks temporal windows "
                "by observable abnormality. "
                "It does not classify collisions "
                "or critical events."
            ),
        },

        "entities_analyzed": len(
            entity_states
        ),

        "entity_states_analyzed": total_states,

        "temporal_windows_analyzed": len(
            windows
        ),

        "top_candidates": ranked_results[
            :TOP_K
        ],

        "ranked_windows": ranked_results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2,
        )

    print_summary(ranked_results)

    print()

    print(
        f"[temporal] Saved analysis to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
