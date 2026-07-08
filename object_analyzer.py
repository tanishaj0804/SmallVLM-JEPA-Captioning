"""
Tracked object motion analyzer.

Reads tracking JSON produced by object_tracker.py.

Analyzes:
- track quality
- trajectory continuity
- raw pixel motion
- bounding-box normalized motion
- depth motion using bounding-box area change
- lateral movement
- motion phases
- motion event flags

Output:
- object analysis JSON for later interaction analysis
"""

import argparse
import json
import math
import os
from collections import Counter


CONFIRMED_MIN_FRAMES = 15
CONFIRMED_MIN_CONTINUITY = 0.70

PROVISIONAL_MIN_FRAMES = 8
PROVISIONAL_MIN_CONTINUITY = 0.50

RAPID_NORMALIZED_SPEED = 1.5
MODERATE_NORMALIZED_SPEED = 0.5

RAPID_APPROACH_GROWTH = 2.0
APPROACH_GROWTH = 1.25

RAPID_RECEDE_RATIO = 0.50
RECEDE_RATIO = 0.80

STRONG_LATERAL_THRESHOLD = 0.20
LATERAL_THRESHOLD = 0.08

PHASE_WINDOW_SECONDS = 0.40


def euclidean_distance(point_a, point_b):
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]

    return math.sqrt(
        dx * dx + dy * dy
    )


def bbox_diagonal(box):
    x1, y1, x2, y2 = box

    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)

    return math.sqrt(
        width * width + height * height
    )


def calculate_continuity(
    detections,
    first_frame,
    last_frame,
):
    expected_frames = (
        last_frame - first_frame + 1
    )

    if expected_frames <= 0:
        return 0.0

    return len(detections) / expected_frames


def calculate_raw_motion(detections):
    if len(detections) < 2:
        return {
            "total_distance_px": 0.0,
            "average_speed_px_per_sec": 0.0,
            "maximum_speed_px_per_sec": 0.0,
        }

    total_distance = 0.0
    speeds = []

    for previous, current in zip(
        detections[:-1],
        detections[1:],
    ):
        time_difference = (
            current["time_seconds"]
            - previous["time_seconds"]
        )

        if time_difference <= 0:
            continue

        distance = euclidean_distance(
            previous["center"],
            current["center"],
        )

        speed = distance / time_difference

        total_distance += distance
        speeds.append(speed)

    if not speeds:
        return {
            "total_distance_px": 0.0,
            "average_speed_px_per_sec": 0.0,
            "maximum_speed_px_per_sec": 0.0,
        }

    return {
        "total_distance_px": round(
            total_distance,
            3,
        ),
        "average_speed_px_per_sec": round(
            sum(speeds) / len(speeds),
            3,
        ),
        "maximum_speed_px_per_sec": round(
            max(speeds),
            3,
        ),
    }


def calculate_normalized_motion(detections):
    if len(detections) < 2:
        return {
            "average_normalized_speed": 0.0,
            "maximum_normalized_speed": 0.0,
            "motion_samples": [],
        }

    normalized_speeds = []
    motion_samples = []

    for previous, current in zip(
        detections[:-1],
        detections[1:],
    ):
        time_difference = (
            current["time_seconds"]
            - previous["time_seconds"]
        )

        if time_difference <= 0:
            continue

        distance = euclidean_distance(
            previous["center"],
            current["center"],
        )

        previous_diagonal = bbox_diagonal(
            previous["bbox"]
        )

        current_diagonal = bbox_diagonal(
            current["bbox"]
        )

        reference_diagonal = (
            previous_diagonal
            + current_diagonal
        ) / 2

        if reference_diagonal <= 0:
            continue

        normalized_distance = (
            distance / reference_diagonal
        )

        normalized_speed = (
            normalized_distance
            / time_difference
        )

        normalized_speeds.append(
            normalized_speed
        )

        motion_samples.append(
            {
                "start_frame": previous["frame"],
                "end_frame": current["frame"],
                "start_time": previous[
                    "time_seconds"
                ],
                "end_time": current[
                    "time_seconds"
                ],
                "normalized_speed": round(
                    normalized_speed,
                    4,
                ),
            }
        )

    if not normalized_speeds:
        return {
            "average_normalized_speed": 0.0,
            "maximum_normalized_speed": 0.0,
            "motion_samples": [],
        }

    return {
        "average_normalized_speed": round(
            sum(normalized_speeds)
            / len(normalized_speeds),
            4,
        ),
        "maximum_normalized_speed": round(
            max(normalized_speeds),
            4,
        ),
        "motion_samples": motion_samples,
    }


def analyze_depth_motion(detections):
    if len(detections) < 2:
        return {
            "initial_area_ratio": 0.0,
            "final_area_ratio": 0.0,
            "area_growth_ratio": 1.0,
            "depth_motion": "unknown",
            "rapid_approach": False,
        }

    area_values = [
        detection.get(
            "area_ratio",
            0.0,
        )
        for detection in detections
        if detection.get(
            "area_ratio",
            0.0,
        ) > 0
    ]

    if len(area_values) < 2:
        return {
            "initial_area_ratio": 0.0,
            "final_area_ratio": 0.0,
            "area_growth_ratio": 1.0,
            "depth_motion": "unknown",
            "rapid_approach": False,
        }

    sample_size = max(
        1,
        min(
            5,
            len(area_values) // 3,
        ),
    )

    initial_area = (
        sum(area_values[:sample_size])
        / sample_size
    )

    final_area = (
        sum(area_values[-sample_size:])
        / sample_size
    )

    if initial_area <= 0:
        growth_ratio = 1.0
    else:
        growth_ratio = (
            final_area / initial_area
        )

    if growth_ratio >= RAPID_APPROACH_GROWTH:
        depth_motion = "rapidly_approaching"

    elif growth_ratio >= APPROACH_GROWTH:
        depth_motion = "approaching"

    elif growth_ratio <= RAPID_RECEDE_RATIO:
        depth_motion = "rapidly_receding"

    elif growth_ratio <= RECEDE_RATIO:
        depth_motion = "receding"

    else:
        depth_motion = "stable_depth"

    return {
        "initial_area_ratio": round(
            initial_area,
            5,
        ),
        "final_area_ratio": round(
            final_area,
            5,
        ),
        "area_growth_ratio": round(
            growth_ratio,
            4,
        ),
        "depth_motion": depth_motion,
        "rapid_approach": (
            growth_ratio
            >= RAPID_APPROACH_GROWTH
        ),
    }


def analyze_lateral_motion(
    detections,
    frame_width,
):
    if (
        len(detections) < 2
        or frame_width <= 0
    ):
        return {
            "horizontal_displacement_px": 0.0,
            "normalized_horizontal_displacement": 0.0,
            "lateral_motion": "unknown",
        }

    start_x = detections[0]["center"][0]
    end_x = detections[-1]["center"][0]

    displacement = end_x - start_x

    normalized_displacement = (
        displacement / frame_width
    )

    if (
        normalized_displacement
        <= -STRONG_LATERAL_THRESHOLD
    ):
        motion = "strong_left_motion"

    elif (
        normalized_displacement
        <= -LATERAL_THRESHOLD
    ):
        motion = "left_motion"

    elif (
        normalized_displacement
        >= STRONG_LATERAL_THRESHOLD
    ):
        motion = "strong_right_motion"

    elif (
        normalized_displacement
        >= LATERAL_THRESHOLD
    ):
        motion = "right_motion"

    else:
        motion = "mostly_longitudinal"

    return {
        "horizontal_displacement_px": round(
            displacement,
            3,
        ),
        "normalized_horizontal_displacement": round(
            normalized_displacement,
            4,
        ),
        "lateral_motion": motion,
    }


def classify_motion_state(
    normalized_speed,
    area_growth,
    horizontal_change,
):
    if area_growth >= RAPID_APPROACH_GROWTH:
        return "rapid_approach"

    if area_growth >= APPROACH_GROWTH:
        return "approaching"

    if area_growth <= RAPID_RECEDE_RATIO:
        return "rapid_recede"

    if area_growth <= RECEDE_RATIO:
        return "receding"

    if (
        abs(horizontal_change)
        >= STRONG_LATERAL_THRESHOLD
    ):
        return "strong_lateral_motion"

    if (
        abs(horizontal_change)
        >= LATERAL_THRESHOLD
    ):
        return "lateral_motion"

    if normalized_speed >= RAPID_NORMALIZED_SPEED:
        return "rapid_motion"

    if (
        normalized_speed
        >= MODERATE_NORMALIZED_SPEED
    ):
        return "normal_motion"

    return "low_motion"


def build_motion_phases(
    detections,
    frame_width,
):
    if len(detections) < 3:
        return []

    start_time = detections[0][
        "time_seconds"
    ]

    end_time = detections[-1][
        "time_seconds"
    ]

    phases = []

    window_start = start_time

    while window_start < end_time:
        window_end = (
            window_start
            + PHASE_WINDOW_SECONDS
        )

        window_detections = [
            detection
            for detection in detections
            if (
                window_start
                <= detection["time_seconds"]
                <= window_end
            )
        ]

        if len(window_detections) < 2:
            window_start = window_end
            continue

        normalized_result = (
            calculate_normalized_motion(
                window_detections
            )
        )

        first_area = window_detections[0].get(
            "area_ratio",
            0.0,
        )

        last_area = window_detections[-1].get(
            "area_ratio",
            0.0,
        )

        if first_area > 0:
            area_growth = (
                last_area / first_area
            )
        else:
            area_growth = 1.0

        first_x = window_detections[0][
            "center"
        ][0]

        last_x = window_detections[-1][
            "center"
        ][0]

        if frame_width > 0:
            horizontal_change = (
                last_x - first_x
            ) / frame_width
        else:
            horizontal_change = 0.0

        state = classify_motion_state(
            normalized_result[
                "average_normalized_speed"
            ],
            area_growth,
            horizontal_change,
        )

        phases.append(
            {
                "start_time": round(
                    window_start,
                    3,
                ),
                "end_time": round(
                    min(
                        window_end,
                        end_time,
                    ),
                    3,
                ),
                "state": state,
                "average_normalized_speed": (
                    normalized_result[
                        "average_normalized_speed"
                    ]
                ),
                "area_growth_ratio": round(
                    area_growth,
                    4,
                ),
                "horizontal_change": round(
                    horizontal_change,
                    4,
                ),
            }
        )

        window_start = window_end

    return merge_motion_phases(phases)


def merge_motion_phases(phases):
    if not phases:
        return []

    merged = [phases[0].copy()]

    for phase in phases[1:]:
        previous = merged[-1]

        if (
            previous["state"]
            == phase["state"]
        ):
            previous["end_time"] = phase[
                "end_time"
            ]

            previous[
                "average_normalized_speed"
            ] = round(
                (
                    previous[
                        "average_normalized_speed"
                    ]
                    + phase[
                        "average_normalized_speed"
                    ]
                )
                / 2,
                4,
            )

            previous[
                "area_growth_ratio"
            ] = round(
                phase["area_growth_ratio"],
                4,
            )

            previous[
                "horizontal_change"
            ] = round(
                previous["horizontal_change"]
                + phase["horizontal_change"],
                4,
            )

        else:
            merged.append(
                phase.copy()
            )

    return merged


def classify_track_quality(
    frames_seen,
    continuity,
):
    if (
        frames_seen >= CONFIRMED_MIN_FRAMES
        and continuity
        >= CONFIRMED_MIN_CONTINUITY
    ):
        return "confirmed"

    if (
        frames_seen >= PROVISIONAL_MIN_FRAMES
        and continuity
        >= PROVISIONAL_MIN_CONTINUITY
    ):
        return "provisional"

    return "noisy"


def build_motion_events(
    normalized_motion,
    depth_motion,
    lateral_motion,
    motion_phases,
):
    events = []

    if depth_motion["rapid_approach"]:
        events.append(
            "rapid_approach"
        )

    if depth_motion["depth_motion"] in (
        "rapidly_receding",
        "receding",
    ):
        events.append(
            depth_motion["depth_motion"]
        )

    if lateral_motion["lateral_motion"] in (
        "strong_left_motion",
        "strong_right_motion",
    ):
        events.append(
            "strong_lateral_displacement"
        )

    if (
        normalized_motion[
            "maximum_normalized_speed"
        ]
        >= RAPID_NORMALIZED_SPEED
    ):
        events.append(
            "high_apparent_motion"
        )

    phase_states = [
        phase["state"]
        for phase in motion_phases
    ]

    if (
        "rapid_motion" in phase_states
        and "low_motion" in phase_states
    ):
        events.append(
            "rapid_to_low_motion_transition"
        )

    if (
        "rapid_approach" in phase_states
        and (
            "lateral_motion" in phase_states
            or "strong_lateral_motion"
            in phase_states
        )
    ):
        events.append(
            "approach_with_lateral_change"
        )

    return list(dict.fromkeys(events))

def analyze_entity(
    entity,
    detections,
    frame_width,
):
    frames_seen = len(detections)

    if detections:
        first_frame = detections[0]["frame"]
        last_frame = detections[-1]["frame"]
    else:
        first_frame = entity.get(
            "first_frame",
            0,
        )
        last_frame = entity.get(
            "last_frame",
            first_frame,
        )

    continuity = calculate_continuity(
        detections,
        first_frame,
        last_frame,
    )

    track_quality = classify_track_quality(
        frames_seen,
        continuity,
    )

    raw_motion = calculate_raw_motion(
        detections
    )

    normalized_motion = (
        calculate_normalized_motion(
            detections
        )
    )

    depth_motion = analyze_depth_motion(
        detections
    )

    lateral_motion = analyze_lateral_motion(
        detections,
        frame_width,
    )

    motion_phases = build_motion_phases(
        detections,
        frame_width,
    )

    motion_events = build_motion_events(
        normalized_motion,
        depth_motion,
        lateral_motion,
        motion_phases,
    )

    trusted_for_interaction = (
        track_quality == "confirmed"
    )

    tracker_id = entity.get(
        "tracker_id"
    )

    source_track_ids = entity.get(
        "source_track_ids",
        [],
    )

    if (
        tracker_id is not None
        and not source_track_ids
    ):
        source_track_ids = [
            tracker_id
        ]

    duration_seconds = 0.0

    if detections:
        duration_seconds = (
            detections[-1]["time_seconds"]
            - detections[0]["time_seconds"]
        )

    return {
        "id": entity.get(
            "id",
            "unknown_entity",
        ),
        "type": entity.get(
            "type",
            "unknown",
        ),
        "tracker_id": tracker_id,
        "source_track_ids": source_track_ids,
        "track_quality": track_quality,
        "trusted_for_interaction": (
            trusted_for_interaction
        ),
        "frames_seen": frames_seen,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "duration_seconds": round(
            duration_seconds,
            3,
        ),
        "continuity": round(
            continuity,
            4,
        ),
        "class_confidence": entity.get(
            "class_confidence",
            0.0,
        ),
        "class_votes": entity.get(
            "class_votes",
            {},
        ),
        "raw_motion": raw_motion,
        "normalized_motion": {
            "average_normalized_speed": (
                normalized_motion[
                    "average_normalized_speed"
                ]
            ),
            "maximum_normalized_speed": (
                normalized_motion[
                    "maximum_normalized_speed"
                ]
            ),
        },
        "depth_motion": depth_motion,
        "lateral_motion": lateral_motion,
        "motion_phases": motion_phases,
        "motion_events": motion_events,
    }

def analyze_tracking(
    tracking_json,
    output_json,
):
    print(
        f"[analysis] Reading: "
        f"{tracking_json}"
    )

    with open(
        tracking_json,
        "r",
        encoding="utf-8",
    ) as file:
        tracking_data = json.load(file)

    metadata = tracking_data[
        "video_metadata"
    ]

    frame_width = metadata["width"]

    entities = tracking_data[
        "tracked_entities"
    ]

    trajectories = tracking_data[
        "trajectories"
    ]

    print(
        "[analysis] Analyzing "
        "tracked entities..."
    )

    analyzed_entities = []

    quality_counts = Counter()

    raw_counts = tracking_data.get(
        "object_counts",
        {},
    )

    confirmed_counts = {
        "cars": 0,
        "buses": 0,
        "motorcycles": 0,
        "bicycles": 0,
        "trucks": 0,
        "people": 0,
    }

    count_keys = {
        "car": "cars",
        "bus": "buses",
        "motorcycle": "motorcycles",
        "bicycle": "bicycles",
        "truck": "trucks",
        "person": "people",
    }

    for entity in entities:
        entity_id = entity["id"]

        detections = trajectories.get(
            entity_id,
            [],
        )

        analysis = analyze_entity(
            entity,
            detections,
            frame_width,
        )

        analyzed_entities.append(
            analysis
        )

        quality = analysis[
            "track_quality"
        ]

        quality_counts[quality] += 1

        if quality == "confirmed":
            entity_type = analysis["type"]

            count_key = count_keys.get(
                entity_type
            )

            if count_key:
                confirmed_counts[
                    count_key
                ] += 1

        print(
            f"  {entity_id:<15} "
            f"{quality:<12} "
            f"frames={analysis['frames_seen']:<4} "
            f"continuity="
            f"{analysis['continuity']:.2f} "
            f"norm_speed="
            f"{analysis['normalized_motion']['average_normalized_speed']:.2f} "
            f"depth="
            f"{analysis['depth_motion']['depth_motion']:<20} "
            f"lateral="
            f"{analysis['lateral_motion']['lateral_motion']}"
        )

    trusted_entities = [
        entity
        for entity in analyzed_entities
        if entity[
            "trusted_for_interaction"
        ]
    ]

    motion_event_summary = Counter()

    for entity in trusted_entities:
        for event in entity[
            "motion_events"
        ]:
            motion_event_summary[event] += 1

    output = {
        "video_metadata": metadata,
        "analysis_config": {
            "confirmed_min_frames": (
                CONFIRMED_MIN_FRAMES
            ),
            "confirmed_min_continuity": (
                CONFIRMED_MIN_CONTINUITY
            ),
            "provisional_min_frames": (
                PROVISIONAL_MIN_FRAMES
            ),
            "provisional_min_continuity": (
                PROVISIONAL_MIN_CONTINUITY
            ),
            "rapid_normalized_speed": (
                RAPID_NORMALIZED_SPEED
            ),
            "rapid_approach_growth": (
                RAPID_APPROACH_GROWTH
            ),
            "phase_window_seconds": (
                PHASE_WINDOW_SECONDS
            ),
        },
        "raw_object_counts": raw_counts,
        "confirmed_object_counts": (
            confirmed_counts
        ),
        "track_quality_summary": dict(
            quality_counts
        ),
        "motion_event_summary": dict(
            motion_event_summary
        ),
        "analyzed_entities": (
            analyzed_entities
        ),
        "trusted_entities": [
            entity["id"]
            for entity in trusted_entities
        ],
    }

    os.makedirs(
        os.path.dirname(output_json) or ".",
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

    print()
    print("=" * 72)
    print("OBJECT ANALYSIS COMPLETE")
    print("=" * 72)

    print("\nTrack quality:")

    for quality in (
        "confirmed",
        "provisional",
        "noisy",
    ):
        print(
            f"  {quality:<15} "
            f"{quality_counts[quality]}"
        )

    print("\nRaw stitched object counts:")

    for key, value in raw_counts.items():
        print(
            f"  {key:<15} {value}"
        )

    print("\nConfirmed object counts:")

    for key, value in (
        confirmed_counts.items()
    ):
        print(
            f"  {key:<15} {value}"
        )

    print("\nTrusted motion events:")

    if motion_event_summary:
        for event, count in (
            motion_event_summary.items()
        ):
            print(
                f"  {event:<35} {count}"
            )
    else:
        print(
            "  No strong motion events detected"
        )

    print(
        f"\nAnalysis written to: "
        f"{output_json}"
    )

    print("=" * 72)

    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze tracked traffic "
            "object trajectories"
        )
    )

    parser.add_argument(
        "tracking_json",
        help=(
            "Tracking JSON produced by "
            "object_tracker.py"
        ),
    )

    parser.add_argument(
        "--output_json",
        default=None,
        help="Output analysis JSON",
    )

    args = parser.parse_args()

    input_name = os.path.basename(
        args.tracking_json
    )

    video_name = input_name.replace(
        "_tracking.json",
        "",
    )

    output_json = (
        args.output_json
        or (
            f"results/"
            f"{video_name}_object_analysis.json"
        )
    )

    analyze_tracking(
        args.tracking_json,
        output_json,
    )


if __name__ == "__main__":
    main()
