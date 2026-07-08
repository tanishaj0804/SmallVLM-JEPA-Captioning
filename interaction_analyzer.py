import argparse
import json
import math
import os
from collections import Counter


CLASS_SIZE_PRIORS = {
    "person": 1.0,
    "bicycle": 1.2,
    "motorcycle": 1.5,
    "car": 2.8,
    "truck": 4.5,
    "bus": 5.0,
}

VULNERABLE_TYPES = {
    "person",
    "bicycle",
    "motorcycle",
}

VEHICLE_TYPES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
}

MIN_SHARED_FRAMES = 5
EVENT_RADIUS = 8

TRAFFIC_CONFLICT_THRESHOLD = 0.40
ACCIDENT_THRESHOLD = 0.58

MAX_NORMALIZED_DISTANCE = 1.5


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def euclidean(point_a, point_b):
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2
        + (point_a[1] - point_b[1]) ** 2
    )


def bbox_area(box):
    x1, y1, x2, y2 = box

    return max(0.0, x2 - x1) * max(
        0.0,
        y2 - y1,
    )


def bbox_diagonal(box):
    x1, y1, x2, y2 = box

    return math.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
    )


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)

    union_area = (
        area_a + area_b - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def normalized_center_distance(
    detection_a,
    detection_b,
):
    center_distance = euclidean(
        detection_a["center"],
        detection_b["center"],
    )

    diagonal_a = bbox_diagonal(
        detection_a["bbox"]
    )

    diagonal_b = bbox_diagonal(
        detection_b["bbox"]
    )

    reference_scale = max(
        1.0,
        (diagonal_a + diagonal_b) / 2.0,
    )

    return center_distance / reference_scale


def projected_scale(detection, entity_type):
    area = bbox_area(detection["bbox"])

    prior = CLASS_SIZE_PRIORS.get(
        entity_type,
        2.0,
    )

    return math.sqrt(max(area, 1.0)) / prior


def depth_consistency(
    detection_a,
    detection_b,
    type_a,
    type_b,
):
    scale_a = projected_scale(
        detection_a,
        type_a,
    )

    scale_b = projected_scale(
        detection_b,
        type_b,
    )

    larger = max(scale_a, scale_b)
    smaller = min(scale_a, scale_b)

    if larger <= 0:
        return 0.0

    ratio = smaller / larger

    return clamp(ratio)


def build_detection_map(detections):
    return {
        detection["frame"]: detection
        for detection in detections
    }


def get_shared_frames(
    detections_a,
    detections_b,
):
    frames_a = set(detections_a.keys())
    frames_b = set(detections_b.keys())

    return sorted(frames_a & frames_b)


def get_velocity(
    detection_map,
    frame,
    previous=True,
    window=3,
):
    if previous:
        candidate_frames = [
            frame - offset
            for offset in range(1, window + 1)
        ]
    else:
        candidate_frames = [
            frame + offset
            for offset in range(1, window + 1)
        ]

    current = detection_map.get(frame)

    if current is None:
        return None

    for other_frame in candidate_frames:
        other = detection_map.get(other_frame)

        if other is None:
            continue

        if previous:
            dx = (
                current["center"][0]
                - other["center"][0]
            )

            dy = (
                current["center"][1]
                - other["center"][1]
            )

            frame_delta = frame - other_frame
        else:
            dx = (
                other["center"][0]
                - current["center"][0]
            )

            dy = (
                other["center"][1]
                - current["center"][1]
            )

            frame_delta = other_frame - frame

        if frame_delta <= 0:
            continue

        return [
            dx / frame_delta,
            dy / frame_delta,
        ]

    return None


def vector_magnitude(vector):
    if vector is None:
        return 0.0

    return math.sqrt(
        vector[0] ** 2 + vector[1] ** 2
    )


def get_local_velocity(
    detection_map,
    start_frame,
    end_frame,
):
    available_frames = sorted(
        frame
        for frame in detection_map
        if start_frame <= frame <= end_frame
    )

    if len(available_frames) < 2:
        return None

    first_frame = available_frames[0]
    last_frame = available_frames[-1]

    first_detection = detection_map[first_frame]
    last_detection = detection_map[last_frame]

    frame_delta = last_frame - first_frame

    if frame_delta <= 0:
        return None

    dx = (
        last_detection["center"][0]
        - first_detection["center"][0]
    )

    dy = (
        last_detection["center"][1]
        - first_detection["center"][1]
    )

    return [
        dx / frame_delta,
        dy / frame_delta,
    ]


def cosine_similarity(vector_a, vector_b):
    magnitude_a = vector_magnitude(vector_a)
    magnitude_b = vector_magnitude(vector_b)

    if magnitude_a <= 0 or magnitude_b <= 0:
        return 1.0

    dot_product = (
        vector_a[0] * vector_b[0]
        + vector_a[1] * vector_b[1]
    )

    return clamp(
        (
            dot_product
            / (magnitude_a * magnitude_b)
            + 1.0
        )
        / 2.0
    )


def motion_response_score(
    detection_map,
    event_frame,
    radius=6,
):
    pre_velocity = get_local_velocity(
        detection_map,
        event_frame - radius,
        event_frame - 1,
    )

    post_velocity = get_local_velocity(
        detection_map,
        event_frame + 1,
        event_frame + radius,
    )

    if (
        pre_velocity is None
        or post_velocity is None
    ):
        return {
            "response_score": 0.0,
            "direction_change": 0.0,
            "speed_change": 0.0,
        }

    pre_speed = vector_magnitude(pre_velocity)
    post_speed = vector_magnitude(post_velocity)

    direction_similarity = cosine_similarity(
        pre_velocity,
        post_velocity,
    )

    direction_change = (
        1.0 - direction_similarity
    )

    speed_reference = max(
        pre_speed,
        post_speed,
        1.0,
    )

    speed_change = clamp(
        abs(pre_speed - post_speed)
        / speed_reference
    )

    response_score = clamp(
        0.65 * direction_change
        + 0.35 * speed_change
    )

    return {
        "response_score": response_score,
        "direction_change": direction_change,
        "speed_change": speed_change,
    }
def orientation(point_a, point_b, point_c):
    value = (
        (point_b[1] - point_a[1])
        * (point_c[0] - point_b[0])
        - (
            point_b[0] - point_a[0]
        )
        * (point_c[1] - point_b[1])
    )

    if abs(value) < 1e-6:
        return 0

    return 1 if value > 0 else 2


def segments_intersect(
    p1,
    q1,
    p2,
    q2,
):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    return (
        o1 != o2
        and o3 != o4
    )


def trajectory_crossing_score(
    map_a,
    map_b,
    event_frame,
    radius=8,
):
    frame_a_before = event_frame - radius
    frame_a_after = event_frame + radius

    frames_a = sorted(
        frame
        for frame in map_a
        if (
            frame_a_before
            <= frame
            <= frame_a_after
        )
    )

    frames_b = sorted(
        frame
        for frame in map_b
        if (
            frame_a_before
            <= frame
            <= frame_a_after
        )
    )

    if (
        len(frames_a) < 2
        or len(frames_b) < 2
    ):
        return 0.0

    a_start = map_a[
        frames_a[0]
    ]["center"]

    a_end = map_a[
        frames_a[-1]
    ]["center"]

    b_start = map_b[
        frames_b[0]
    ]["center"]

    b_end = map_b[
        frames_b[-1]
    ]["center"]

    if segments_intersect(
        a_start,
        a_end,
        b_start,
        b_end,
    ):
        return 1.0

    return 0.0

def distance_series(
    shared_frames,
    map_a,
    map_b,
):
    series = []

    for frame in shared_frames:
        detection_a = map_a[frame]
        detection_b = map_b[frame]

        distance = normalized_center_distance(
            detection_a,
            detection_b,
        )

        iou = bbox_iou(
            detection_a["bbox"],
            detection_b["bbox"],
        )

        series.append(
            {
                "frame": frame,
                "distance": distance,
                "iou": iou,
            }
        )

    return series


def moving_average(values, radius=2):
    output = []

    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(
            len(values),
            index + radius + 1,
        )

        window = values[start:end]

        output.append(
            sum(window) / len(window)
        )

    return output


def find_peak_event(distance_data):
    raw_distances = [
        item["distance"]
        for item in distance_data
    ]

    smoothed = moving_average(
        raw_distances,
        radius=2,
    )

    peak_index = min(
        range(len(smoothed)),
        key=lambda index: smoothed[index],
    )

    return peak_index, smoothed


def get_event_window(
    distance_data,
    peak_index,
):
    start_index = max(
        0,
        peak_index - EVENT_RADIUS,
    )

    end_index = min(
        len(distance_data) - 1,
        peak_index + EVENT_RADIUS,
    )

    return start_index, end_index


def phase_average(
    values,
    start,
    end,
):
    if start > end:
        return None

    phase = values[start:end + 1]

    if not phase:
        return None

    return sum(phase) / len(phase)


def analyze_distance_phases(
    smoothed_distances,
    start_index,
    peak_index,
    end_index,
):
    pre_start = max(
        0,
        start_index - EVENT_RADIUS,
    )

    pre_end = start_index - 1

    post_start = end_index + 1

    post_end = min(
        len(smoothed_distances) - 1,
        end_index + EVENT_RADIUS,
    )

    pre_distance = phase_average(
        smoothed_distances,
        pre_start,
        pre_end,
    )

    event_distance = phase_average(
        smoothed_distances,
        start_index,
        end_index,
    )

    post_distance = phase_average(
        smoothed_distances,
        post_start,
        post_end,
    )

    peak_distance = smoothed_distances[
        peak_index
    ]

    convergence_score = 0.0

    if (
        pre_distance is not None
        and event_distance is not None
        and pre_distance > 0
    ):
        convergence_score = clamp(
            (
                pre_distance - event_distance
            )
            / pre_distance
        )

    separation_score = 0.0

    if (
        post_distance is not None
        and event_distance is not None
        and event_distance > 0
    ):
        separation_score = clamp(
            (
                post_distance - event_distance
            )
            / event_distance
        )

    return {
        "pre_distance": pre_distance,
        "event_distance": event_distance,
        "post_distance": post_distance,
        "peak_distance": peak_distance,
        "convergence_score": convergence_score,
        "separation_score": separation_score,
    }


def contact_pattern_score(
    distance_data,
    start_index,
    end_index,
):
    event_items = distance_data[
        start_index:end_index + 1
    ]

    if not event_items:
        return {
            "contact_ratio": 0.0,
            "contact_pattern_score": 0.0,
            "persistent_overlap_penalty": 0.0,
        }

    contact_flags = []

    for item in event_items:
        is_contact = (
            item["distance"] <= 0.65
            or item["iou"] >= 0.08
        )

        contact_flags.append(is_contact)

    contact_ratio = (
        sum(contact_flags)
        / len(contact_flags)
    )

    # Short, localized contact is more accident-like.
    if 0.10 <= contact_ratio <= 0.55:
        pattern_score = 1.0

    elif contact_ratio < 0.10:
        pattern_score = (
            contact_ratio / 0.10
        )

    else:
        pattern_score = clamp(
            1.0
            - (
                contact_ratio - 0.55
            )
            / 0.45
        )

    persistent_overlap_penalty = 0.0

    if contact_ratio >= 0.75:
        persistent_overlap_penalty = clamp(
            (
                contact_ratio - 0.75
            )
            / 0.25
        )

    return {
        "contact_ratio": contact_ratio,
        "contact_pattern_score": (
            pattern_score
        ),
        "persistent_overlap_penalty": (
            persistent_overlap_penalty
        ),
    }


def average_depth_consistency(
    frames,
    map_a,
    map_b,
    type_a,
    type_b,
):
    if not frames:
        return 0.0

    values = []

    for frame in frames:
        values.append(
            depth_consistency(
                map_a[frame],
                map_b[frame],
                type_a,
                type_b,
            )
        )

    return sum(values) / len(values)


def is_vulnerable_pair(type_a, type_b):
    return (
        (
            type_a in VEHICLE_TYPES
            and type_b in VULNERABLE_TYPES
        )
        or (
            type_b in VEHICLE_TYPES
            and type_a in VULNERABLE_TYPES
        )
    )


def is_vehicle_vehicle_pair(type_a, type_b):
    return (
        type_a in VEHICLE_TYPES
        and type_b in VEHICLE_TYPES
    )


def get_entity_analysis(
    analysis_lookup,
    entity_id,
):
    return analysis_lookup.get(
        entity_id,
        {},
    )


def has_motion_event(
    entity_analysis,
    event_name,
):
    events = entity_analysis.get(
        "motion_events",
        []
    )

    for event in events:
        if isinstance(event, str):
            if event == event_name:
                return True

        elif isinstance(event, dict):
            if (
                event.get("event")
                == event_name
            ):
                return True

    return False


def calculate_pair_score(
    type_a,
    type_b,
    phase_data,
    depth_score,
    contact_data,
    response_data_a,
    response_data_b,
    crossing_score,
    analysis_a,
    analysis_b,
):
    convergence = phase_data["convergence_score"]
    separation = phase_data["separation_score"]
    peak_distance = phase_data["peak_distance"]

    proximity_score = clamp(
        1.0 - (peak_distance / MAX_NORMALIZED_DISTANCE)
    )

    response_a = response_data_a["response_score"]
    response_b = response_data_b["response_score"]
    post_response = max(response_a, response_b)
    response_asymmetry = abs(response_a - response_b)

    contact_pattern = contact_data["contact_pattern_score"]
    contact_ratio = contact_data["contact_ratio"]
    persistent_overlap_penalty = contact_data[
        "persistent_overlap_penalty"
    ]

    vulnerable_pair = is_vulnerable_pair(type_a, type_b)
    vehicle_vehicle = is_vehicle_vehicle_pair(type_a, type_b)

    if vulnerable_pair:
        effective_depth_score = max(
            depth_score,
            proximity_score * 0.70,
        )
    else:
        effective_depth_score = depth_score

    score = (
        0.20 * convergence
        + 0.15 * proximity_score
        + 0.15 * contact_pattern
        + 0.10 * effective_depth_score
        + 0.15 * post_response
        + 0.15 * response_asymmetry
        + 0.10 * crossing_score
    )

    vulnerable_response = 0.0

    if type_a in VULNERABLE_TYPES:
        vulnerable_response = max(
            vulnerable_response,
            response_a,
        )

    if type_b in VULNERABLE_TYPES:
        vulnerable_response = max(
            vulnerable_response,
            response_b,
        )

    vulnerable_bonus = 0.0

    if vulnerable_pair:
        vulnerable_bonus += 0.08

        if vulnerable_response >= 0.45:
            vulnerable_bonus += 0.08

        if response_asymmetry >= 0.35:
            vulnerable_bonus += 0.06

        if crossing_score >= 0.5:
            vulnerable_bonus += 0.05

    score += vulnerable_bonus

    ordinary_overlap_penalty = 0.0

    if vehicle_vehicle:
        ordinary_overlap_penalty += (
            0.25 * persistent_overlap_penalty
        )

        if (
            contact_ratio >= 0.75
            and convergence < 0.30
            and response_asymmetry < 0.25
        ):
            ordinary_overlap_penalty += 0.15

        if (
            crossing_score < 0.5
            and response_asymmetry < 0.20
            and separation < 0.20
        ):
            ordinary_overlap_penalty += 0.10

    score -= ordinary_overlap_penalty

    motion_event_bonus = 0.0
    important_events = {
        "rapid_to_low_motion_transition",
        "rapid_approach",
    }

    for event in important_events:
        if (
            has_motion_event(analysis_a, event)
            or has_motion_event(analysis_b, event)
        ):
            motion_event_bonus += 0.03

    motion_event_bonus = min(motion_event_bonus, 0.06)
    score += motion_event_bonus

    return {
        "score": clamp(score),
        "proximity_score": proximity_score,
        "effective_depth_score": effective_depth_score,
        "post_response_score": post_response,
        "response_asymmetry": response_asymmetry,
        "crossing_score": crossing_score,
        "contact_ratio": contact_ratio,
        "contact_pattern_score": contact_pattern,
        "persistent_overlap_penalty": persistent_overlap_penalty,
        "ordinary_overlap_penalty": ordinary_overlap_penalty,
        "motion_event_bonus": motion_event_bonus,
        "vulnerable_bonus": vulnerable_bonus,
        "vulnerable_response": vulnerable_response,
        "vulnerable_pair": vulnerable_pair,
    }


def classify_event(score, vulnerable_pair):
    if score >= ACCIDENT_THRESHOLD:
        if vulnerable_pair:
            return (
                "possible_vulnerable_road_user_"
                "accident"
            )

        return "possible_accident_event"

    if score >= TRAFFIC_CONFLICT_THRESHOLD:
        if vulnerable_pair:
            return (
                "vulnerable_road_user_conflict"
            )

        return "traffic_conflict"

    return "normal_road_interaction"


def analyze_pair(
    entity_a,
    entity_b,
    trajectories,
    analysis_lookup,
    fps,
):
    id_a = entity_a["id"]
    id_b = entity_b["id"]
    type_a = entity_a["type"]
    type_b = entity_b["type"]

    detections_a = trajectories.get(id_a, [])
    detections_b = trajectories.get(id_b, [])

    map_a = build_detection_map(detections_a)
    map_b = build_detection_map(detections_b)

    shared_frames = get_shared_frames(map_a, map_b)

    if len(shared_frames) < MIN_SHARED_FRAMES:
        return None

    distances = distance_series(
        shared_frames,
        map_a,
        map_b,
    )

    peak_index, smoothed_distances = find_peak_event(
        distances
    )

    start_index, end_index = get_event_window(
        distances,
        peak_index,
    )

    phase_data = analyze_distance_phases(
        smoothed_distances,
        start_index,
        peak_index,
        end_index,
    )

    event_frames = shared_frames[
        start_index:end_index + 1
    ]
    peak_frame = shared_frames[peak_index]

    depth_score = average_depth_consistency(
        event_frames,
        map_a,
        map_b,
        type_a,
        type_b,
    )

    contact_data = contact_pattern_score(
        distances,
        start_index,
        end_index,
    )

    response_data_a = motion_response_score(
        map_a,
        peak_frame,
    )
    response_data_b = motion_response_score(
        map_b,
        peak_frame,
    )

    crossing_score = trajectory_crossing_score(
        map_a,
        map_b,
        peak_frame,
        radius=EVENT_RADIUS,
    )

    analysis_a = get_entity_analysis(
        analysis_lookup,
        id_a,
    )
    analysis_b = get_entity_analysis(
        analysis_lookup,
        id_b,
    )

    score_data = calculate_pair_score(
        type_a,
        type_b,
        phase_data,
        depth_score,
        contact_data,
        response_data_a,
        response_data_b,
        crossing_score,
        analysis_a,
        analysis_b,
    )

    event = classify_event(
        score_data["score"],
        score_data["vulnerable_pair"],
    )

    start_frame = event_frames[0]
    end_frame = event_frames[-1]

    return {
        "entity_a": id_a,
        "entity_b": id_b,
        "type_a": type_a,
        "type_b": type_b,
        "shared_frames": len(shared_frames),
        "event_window": {
            "start_frame": start_frame,
            "peak_frame": peak_frame,
            "end_frame": end_frame,
            "start_time": round(start_frame / fps, 3),
            "peak_time": round(peak_frame / fps, 3),
            "end_time": round(end_frame / fps, 3),
        },
        "phase_analysis": {
            "pre_event_distance": (
                None
                if phase_data["pre_distance"] is None
                else round(phase_data["pre_distance"], 4)
            ),
            "event_distance": (
                None
                if phase_data["event_distance"] is None
                else round(phase_data["event_distance"], 4)
            ),
            "post_event_distance": (
                None
                if phase_data["post_distance"] is None
                else round(phase_data["post_distance"], 4)
            ),
            "peak_distance": round(
                phase_data["peak_distance"],
                4,
            ),
            "convergence_score": round(
                phase_data["convergence_score"],
                4,
            ),
            "separation_score": round(
                phase_data["separation_score"],
                4,
            ),
        },
        "contact_evidence": {
            "contact_ratio": round(
                score_data["contact_ratio"],
                4,
            ),
            "contact_pattern_score": round(
                score_data["contact_pattern_score"],
                4,
            ),
            "persistent_overlap_penalty": round(
                score_data["persistent_overlap_penalty"],
                4,
            ),
            "depth_consistency": round(
                depth_score,
                4,
            ),
            "effective_depth_score": round(
                score_data["effective_depth_score"],
                4,
            ),
            "trajectory_crossing_score": round(
                score_data["crossing_score"],
                4,
            ),
        },
        "post_event_response": {
            id_a: {
                key: round(value, 4)
                for key, value in response_data_a.items()
            },
            id_b: {
                key: round(value, 4)
                for key, value in response_data_b.items()
            },
            "maximum_response": round(
                score_data["post_response_score"],
                4,
            ),
            "response_asymmetry": round(
                score_data["response_asymmetry"],
                4,
            ),
        },
        "scoring": {
            "proximity_score": round(
                score_data["proximity_score"],
                4,
            ),
            "ordinary_overlap_penalty": round(
                score_data["ordinary_overlap_penalty"],
                4,
            ),
            "motion_event_bonus": round(
                score_data["motion_event_bonus"],
                4,
            ),
            "vulnerable_bonus": round(
                score_data["vulnerable_bonus"],
                4,
            ),
            "vulnerable_response": round(
                score_data["vulnerable_response"],
                4,
            ),
        },
        "vulnerable_road_user_pair": score_data[
            "vulnerable_pair"
        ],
        "accident_relevance": round(
            score_data["score"],
            4,
        ),
        "event": event,
    }


def build_analysis_lookup(object_analysis):
    lookup = {}

    entities = object_analysis.get(
        "analyzed_entities",
        object_analysis.get(
            "entities",
            [],
        ),
    )

    for entity in entities:
        entity_id = entity.get("id")

        if entity_id:
            lookup[entity_id] = entity

    return lookup


def get_trusted_entities(
    tracking_data,
    object_analysis,
):
    quality_lookup = {}

    analysis_entities = object_analysis.get(
        "analyzed_entities",
        object_analysis.get(
            "entities",
            [],
        ),
    )

    for entity in analysis_entities:
        entity_id = entity.get("id")

        quality = entity.get(
            "track_quality",
            entity.get("quality"),
        )

        if entity_id:
            quality_lookup[entity_id] = quality

    trusted = []

    for entity in tracking_data.get(
        "tracked_entities",
        [],
    ):
        quality = quality_lookup.get(
            entity["id"]
        )

        if quality == "confirmed":
            trusted.append(entity)

    return trusted


def analyze_interactions(
    tracking_path,
    analysis_path,
    output_path,
):
    print(
        f"[interaction] Reading tracking: "
        f"{tracking_path}"
    )

    print(
        f"[interaction] Reading analysis: "
        f"{analysis_path}"
    )

    tracking_data = load_json(
        tracking_path
    )

    object_analysis = load_json(
        analysis_path
    )

    fps = tracking_data[
        "video_metadata"
    ]["fps"]

    trajectories = tracking_data.get(
        "trajectories",
        {},
    )

    trusted_entities = get_trusted_entities(
        tracking_data,
        object_analysis,
    )

    analysis_lookup = build_analysis_lookup(
        object_analysis
    )

    candidate_pairs = (
        len(trusted_entities)
        * (len(trusted_entities) - 1)
        // 2
    )

    print()
    print(
        f"[interaction] Trusted entities: "
        f"{len(trusted_entities)}"
    )

    print(
        f"[interaction] Candidate pairs: "
        f"{candidate_pairs}"
    )

    interactions = []

    for index_a in range(
        len(trusted_entities)
    ):
        for index_b in range(
            index_a + 1,
            len(trusted_entities),
        ):
            entity_a = trusted_entities[index_a]
            entity_b = trusted_entities[index_b]

            interaction = analyze_pair(
                entity_a,
                entity_b,
                trajectories,
                analysis_lookup,
                fps,
            )

            if interaction is None:
                continue

            interactions.append(interaction)

            if (
                interaction["accident_relevance"]
                >= TRAFFIC_CONFLICT_THRESHOLD
            ):
                print(
                    f"  "
                    f"{interaction['entity_a']:<12} "
                    f"<-> "
                    f"{interaction['entity_b']:<12} "
                    f"relevance="
                    f"{interaction['accident_relevance']:.2f} "
                    f"depth="
                    f"{interaction['contact_evidence']['depth_consistency']:.2f} "
                    f"response="
                    f"{interaction['post_event_response']['maximum_response']:.2f} "
                    f"asym="
                    f"{interaction['post_event_response']['response_asymmetry']:.2f} "
                    f"cross="
                    f"{interaction['contact_evidence']['trajectory_crossing_score']:.2f} "
                    f"event="
                    f"{interaction['event']}"
                )

    interactions.sort(
        key=lambda item: item[
            "accident_relevance"
        ],
        reverse=True,
    )

    accident_candidates = [
        interaction
        for interaction in interactions
        if interaction["accident_relevance"]
        >= ACCIDENT_THRESHOLD
    ]

    vulnerable_events = [
        interaction
        for interaction in interactions
        if interaction[
            "vulnerable_road_user_pair"
        ]
        and interaction["accident_relevance"]
        >= TRAFFIC_CONFLICT_THRESHOLD
    ]

    most_likely_accident = (
        accident_candidates[0]
        if accident_candidates
        else None
    )

    output = {
        "interaction_config": {
            "minimum_shared_frames": (
                MIN_SHARED_FRAMES
            ),
            "event_radius_frames": EVENT_RADIUS,
            "traffic_conflict_threshold": (
                TRAFFIC_CONFLICT_THRESHOLD
            ),
            "accident_threshold": (
                ACCIDENT_THRESHOLD
            ),
            "class_size_priors": (
                CLASS_SIZE_PRIORS
            ),
        },
        "interaction_summary": {
            "trusted_entities": len(
                trusted_entities
            ),
            "candidate_pairs": candidate_pairs,
            "analyzed_pairs": len(
                interactions
            ),
            "accident_candidates": len(
                accident_candidates
            ),
            "vulnerable_road_user_events": len(
                vulnerable_events
            ),
        },
        "most_likely_accident": (
            most_likely_accident
        ),
        "accident_candidates": (
            accident_candidates
        ),
        "vulnerable_road_user_events": (
            vulnerable_events
        ),
        "all_interactions": interactions,
    }

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True,
    )

    with open(
        output_path,
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
    print(
        "EVENT-WINDOW INTERACTION ANALYSIS COMPLETE"
    )
    print("=" * 72)

    print()
    print(
        f"Trusted entities: "
        f"{len(trusted_entities)}"
    )

    print(
        f"Candidate pairs: "
        f"{candidate_pairs}"
    )

    print(
        f"Analyzed pairs: "
        f"{len(interactions)}"
    )

    print(
        f"Accident candidates: "
        f"{len(accident_candidates)}"
    )

    print(
        f"Vulnerable road user events: "
        f"{len(vulnerable_events)}"
    )

    print()
    print("Top interaction candidates:")

    for interaction in interactions[:10]:
        window = interaction["event_window"]

        print(
            f"  "
            f"{interaction['entity_a']:<12} "
            f"<-> "
            f"{interaction['entity_b']:<12} "
            f"relevance="
            f"{interaction['accident_relevance']:.2f} "
            f"depth="
            f"{interaction['contact_evidence']['depth_consistency']:.2f} "
            f"response="
            f"{interaction['post_event_response']['maximum_response']:.2f} "
            f"asym="
            f"{interaction['post_event_response']['response_asymmetry']:.2f} "
            f"cross="
            f"{interaction['contact_evidence']['trajectory_crossing_score']:.2f} "
            f"frames="
            f"{window['start_frame']}-"
            f"{window['peak_frame']}-"
            f"{window['end_frame']} "
            f"{interaction['event']}"
        )

    if most_likely_accident is not None:
        window = most_likely_accident[
            "event_window"
        ]

        print()
        print("MOST LIKELY ACCIDENT EVENT")

        print(
            f"  "
            f"{most_likely_accident['entity_a']} "
            f"<-> "
            f"{most_likely_accident['entity_b']}"
        )

        print(
            f"  Relevance: "
            f"{most_likely_accident['accident_relevance']:.3f}"
        )

        print(
            f"  Event window: "
            f"{window['start_frame']} -> "
            f"{window['peak_frame']} -> "
            f"{window['end_frame']}"
        )

        print(
            f"  Event time: "
            f"{window['start_time']:.3f}s -> "
            f"{window['peak_time']:.3f}s -> "
            f"{window['end_time']:.3f}s"
        )

        print(
            f"  Depth consistency: "
            f"{most_likely_accident['contact_evidence']['depth_consistency']:.3f}"
        )

        print(
            f"  Contact ratio: "
            f"{most_likely_accident['contact_evidence']['contact_ratio']:.3f}"
        )

        print(
            f"  Contact pattern score: "
            f"{most_likely_accident['contact_evidence']['contact_pattern_score']:.3f}"
        )

        print(
            f"  Persistent overlap penalty: "
            f"{most_likely_accident['contact_evidence']['persistent_overlap_penalty']:.3f}"
        )

        print(
            f"  Response asymmetry: "
            f"{most_likely_accident['post_event_response']['response_asymmetry']:.3f}"
        )

        print(
            f"  Trajectory crossing score: "
            f"{most_likely_accident['contact_evidence']['trajectory_crossing_score']:.3f}"
        )

        print(
            f"  Event: "
            f"{most_likely_accident['event']}"
        )

    print()
    print(
        f"Interaction analysis written to: "
        f"{output_path}"
    )

    print("=" * 72)

    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Event-window traffic interaction analyzer"
        )
    )

    parser.add_argument(
        "tracking_json",
        help="Tracking JSON from object_tracker.py",
    )

    parser.add_argument(
        "object_analysis_json",
        help=(
            "Object analysis JSON from "
            "object_analyzer.py"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output interaction JSON",
    )

    args = parser.parse_args()

    video_name = os.path.basename(
        args.tracking_json
    )

    video_name = video_name.replace(
        "_tracking.json",
        "",
    )

    output_path = (
        args.output
        or f"results/{video_name}_interactions.json"
    )

    analyze_interactions(
        args.tracking_json,
        args.object_analysis_json,
        output_path,
    )


if __name__ == "__main__":
    main()