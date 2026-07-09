"""
Traffic object detection and tracking using YOLO + ByteTrack.

Features:
    - Track-level stable class voting
    - Detection rejection
    - FPS-aware track stitching
    - Motion prediction
    - Spatial continuity analysis
    - Area consistency analysis
    - Direction consistency analysis
    - One-to-one track chain stitching
    - Entity quality scoring
    - Weak entity rejection
"""

import argparse
import json
import math
import os

from collections import (
    Counter,
    defaultdict,
)

import cv2

from ultralytics import YOLO


MODEL_NAME = "yolo11n.pt"


TRACKED_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


COUNT_KEYS = {
    "person": "people",
    "bicycle": "bicycles",
    "car": "cars",
    "motorcycle": "motorcycles",
    "bus": "buses",
    "truck": "trucks",
}


VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
}


TWO_WHEELER_CLASSES = {
    "bicycle",
    "motorcycle",
}


# ---------------------------------------------------------
# TRACK VALIDATION
# ---------------------------------------------------------

MIN_TRACK_DURATION_SECONDS = 0.35

MIN_TRACK_FRAMES_ABSOLUTE = 8

MIN_AVERAGE_DETECTION_CONFIDENCE = 0.30


# ---------------------------------------------------------
# ENTITY VALIDATION
# ---------------------------------------------------------

MIN_ENTITY_DURATION_SECONDS = 0.40

MIN_ENTITY_FRAMES = 10

MIN_ENTITY_QUALITY_SCORE = 0.32


# ---------------------------------------------------------
# DETECTION FILTERING
# ---------------------------------------------------------

MAX_BOX_AREA_RATIO = 0.35

DASHCAM_AREA_RATIO = 0.20

DASHCAM_Y_THRESHOLD = 0.60


# ---------------------------------------------------------
# STITCHING
# ---------------------------------------------------------

MAX_STITCH_GAP_SECONDS = 1.50

MAX_STITCH_DISTANCE_RATIO = 0.18

MAX_AREA_CHANGE_RATIO = 4.0

MIN_DIRECTION_SIMILARITY = -0.10

MIN_STITCH_SCORE = 0.60


# ---------------------------------------------------------
# MOTION
# ---------------------------------------------------------

MOTION_WINDOW = 7

MIN_MOTION_MAGNITUDE = 1e-6


def round_box(box):
    return [
        round(float(value), 2)
        for value in box
    ]


def box_center(box):
    x1, y1, x2, y2 = box

    return [
        round(
            (x1 + x2) / 2,
            2,
        ),
        round(
            (y1 + y2) / 2,
            2,
        ),
    ]


def get_area_ratio(
    box,
    frame_width,
    frame_height,
):
    x1, y1, x2, y2 = box

    box_width = max(
        0,
        x2 - x1,
    )

    box_height = max(
        0,
        y2 - y1,
    )

    box_area = (
        box_width * box_height
    )

    frame_area = (
        frame_width * frame_height
    )

    if frame_area <= 0:
        return 0.0

    return (
        box_area / frame_area
    )


def reject_detection(
    box,
    frame_width,
    frame_height,
):
    x1, y1, x2, y2 = box

    area_ratio = get_area_ratio(
        box,
        frame_width,
        frame_height,
    )

    if (
        area_ratio
        > MAX_BOX_AREA_RATIO
    ):
        return (
            True,
            "giant_box",
        )

    if (
        y1
        > frame_height
        * DASHCAM_Y_THRESHOLD
        and area_ratio
        > DASHCAM_AREA_RATIO
    ):
        return (
            True,
            "dashcam_foreground",
        )

    return (
        False,
        None,
    )


def get_stable_class(
    class_history,
):
    if not class_history:
        return None

    counts = Counter(
        class_history
    )

    stable_class, votes = (
        counts.most_common(1)[0]
    )

    return (
        stable_class,
        votes,
        counts,
    )


def same_object_family(
    class_a,
    class_b,
):
    if class_a == class_b:
        return True

    if (
        class_a in VEHICLE_CLASSES
        and class_b in VEHICLE_CLASSES
    ):
        return True

    if (
        class_a
        in TWO_WHEELER_CLASSES
        and class_b
        in TWO_WHEELER_CLASSES
    ):
        return True

    return False


def center_distance(
    center_a,
    center_b,
):
    dx = (
        center_b[0]
        - center_a[0]
    )

    dy = (
        center_b[1]
        - center_a[1]
    )

    return math.sqrt(
        dx * dx
        + dy * dy
    )


def get_motion_vector(
    detections,
    use_end=True,
    window=MOTION_WINDOW,
):
    if len(detections) < 2:
        return None

    if use_end:
        selected = detections[
            -window:
        ]

    else:
        selected = detections[
            :window
        ]

    if len(selected) < 2:
        return None

    first = selected[0]

    last = selected[-1]

    frame_gap = (
        last["frame"]
        - first["frame"]
    )

    if frame_gap <= 0:
        return None

    return (
        (
            last["center"][0]
            - first["center"][0]
        )
        / frame_gap,
        (
            last["center"][1]
            - first["center"][1]
        )
        / frame_gap,
    )


def cosine_similarity(
    vector_a,
    vector_b,
):
    if (
        vector_a is None
        or vector_b is None
    ):
        return None

    ax, ay = vector_a

    bx, by = vector_b

    magnitude_a = math.sqrt(
        ax * ax
        + ay * ay
    )

    magnitude_b = math.sqrt(
        bx * bx
        + by * by
    )

    if (
        magnitude_a
        < MIN_MOTION_MAGNITUDE
        or magnitude_b
        < MIN_MOTION_MAGNITUDE
    ):
        return None

    return (
        ax * bx
        + ay * by
    ) / (
        magnitude_a
        * magnitude_b
    )


def get_area_change_ratio(
    detection_a,
    detection_b,
):
    area_a = detection_a[
        "area_ratio"
    ]

    area_b = detection_b[
        "area_ratio"
    ]

    if (
        area_a <= 0
        or area_b <= 0
    ):
        return float("inf")

    return max(
        area_a / area_b,
        area_b / area_a,
    )


def predict_center(
    detections,
    target_frame,
):
    if not detections:
        return None

    last_detection = (
        detections[-1]
    )

    if len(detections) < 2:
        return last_detection[
            "center"
        ]

    selected = detections[
        -MOTION_WINDOW:
    ]

    first = selected[0]

    last = selected[-1]

    frame_difference = (
        last["frame"]
        - first["frame"]
    )

    if frame_difference <= 0:
        return last["center"]

    velocity_x = (
        last["center"][0]
        - first["center"][0]
    ) / frame_difference

    velocity_y = (
        last["center"][1]
        - first["center"][1]
    ) / frame_difference

    future_frames = (
        target_frame
        - last["frame"]
    )

    return [
        (
            last["center"][0]
            + velocity_x
            * future_frames
        ),
        (
            last["center"][1]
            + velocity_y
            * future_frames
        ),
    ]


def tracks_temporally_overlap(
    track_a,
    track_b,
):
    detections_a = track_a[
        "detections"
    ]

    detections_b = track_b[
        "detections"
    ]

    start_a = detections_a[0][
        "frame"
    ]

    end_a = detections_a[-1][
        "frame"
    ]

    start_b = detections_b[0][
        "frame"
    ]

    end_b = detections_b[-1][
        "frame"
    ]

    return not (
        end_a < start_b
        or end_b < start_a
    )


def get_average_confidence(
    detections,
):
    if not detections:
        return 0.0

    return sum(
        detection["confidence"]
        for detection in detections
    ) / len(detections)


def get_average_area_ratio(
    detections,
):
    if not detections:
        return 0.0

    return sum(
        detection["area_ratio"]
        for detection in detections
    ) / len(detections)


def calculate_track_continuity(
    detections,
):
    if len(detections) < 2:
        return 0.0

    first_frame = detections[0][
        "frame"
    ]

    last_frame = detections[-1][
        "frame"
    ]

    possible_frames = (
        last_frame
        - first_frame
        + 1
    )

    if possible_frames <= 0:
        return 0.0

    return min(
        1.0,
        len(detections)
        / possible_frames,
    )


def calculate_stitch_score(
    track_a,
    track_b,
    frame_width,
    frame_height,
    fps,
):
    detections_a = track_a[
        "detections"
    ]

    detections_b = track_b[
        "detections"
    ]

    if tracks_temporally_overlap(
        track_a,
        track_b,
    ):
        return None

    last_a = detections_a[-1]

    first_b = detections_b[0]

    frame_gap = (
        first_b["frame"]
        - last_a["frame"]
    )

    if frame_gap <= 0:
        return None

    max_stitch_frames = max(
        1,
        int(
            fps
            * MAX_STITCH_GAP_SECONDS
        ),
    )

    if (
        frame_gap
        > max_stitch_frames
    ):
        return None

    if not same_object_family(
        track_a["stable_class"],
        track_b["stable_class"],
    ):
        return None

    predicted_center = predict_center(
        detections_a,
        first_b["frame"],
    )

    if predicted_center is None:
        return None

    distance = center_distance(
        predicted_center,
        first_b["center"],
    )

    frame_diagonal = math.sqrt(
        frame_width * frame_width
        + frame_height * frame_height
    )

    if frame_diagonal <= 0:
        return None

    distance_ratio = (
        distance / frame_diagonal
    )

    if (
        distance_ratio
        > MAX_STITCH_DISTANCE_RATIO
    ):
        return None

    area_change_ratio = (
        get_area_change_ratio(
            last_a,
            first_b,
        )
    )

    if (
        area_change_ratio
        > MAX_AREA_CHANGE_RATIO
    ):
        return None

    motion_a = get_motion_vector(
        detections_a,
        use_end=True,
    )

    motion_b = get_motion_vector(
        detections_b,
        use_end=False,
    )

    direction_similarity = (
        cosine_similarity(
            motion_a,
            motion_b,
        )
    )

    if (
        direction_similarity
        is not None
        and direction_similarity
        < MIN_DIRECTION_SIMILARITY
    ):
        return None

    gap_score = 1.0 - (
        frame_gap
        / max_stitch_frames
    )

    gap_score = max(
        0.0,
        min(
            1.0,
            gap_score,
        ),
    )

    distance_score = 1.0 - (
        distance_ratio
        / MAX_STITCH_DISTANCE_RATIO
    )

    distance_score = max(
        0.0,
        min(
            1.0,
            distance_score,
        ),
    )

    area_score = (
        1.0
        / area_change_ratio
    )

    if direction_similarity is None:
        direction_score = 0.50

    else:
        direction_score = (
            direction_similarity
            + 1.0
        ) / 2.0

    if (
        track_a["stable_class"]
        == track_b["stable_class"]
    ):
        class_score = 1.0

    else:
        class_score = 0.55

    continuity_a = (
        calculate_track_continuity(
            detections_a
        )
    )

    continuity_b = (
        calculate_track_continuity(
            detections_b
        )
    )

    continuity_score = (
        continuity_a
        + continuity_b
    ) / 2.0

    score = (
        0.18 * gap_score
        + 0.32 * distance_score
        + 0.13 * area_score
        + 0.18 * direction_score
        + 0.12 * class_score
        + 0.07 * continuity_score
    )

    return {
        "score": round(
            score,
            4,
        ),
        "frame_gap": frame_gap,
        "gap_seconds": round(
            frame_gap / fps,
            3,
        ),
        "distance_ratio": round(
            distance_ratio,
            4,
        ),
        "area_change_ratio": round(
            area_change_ratio,
            4,
        ),
        "direction_similarity": (
            None
            if direction_similarity
            is None
            else round(
                direction_similarity,
                4,
            )
        ),
        "continuity_score": round(
            continuity_score,
            4,
        ),
    }


def merge_track_group(
    track_ids,
    valid_tracks,
):
    all_detections = []

    all_classes = []

    rejected_detections = 0

    for track_id in track_ids:
        track = valid_tracks[
            track_id
        ]

        all_detections.extend(
            track["detections"]
        )

        for (
            class_name,
            count,
        ) in track[
            "class_votes"
        ].items():
            all_classes.extend(
                [class_name] * count
            )

        rejected_detections += (
            track[
                "rejected_detections"
            ]
        )

    all_detections.sort(
        key=lambda detection: (
            detection["frame"],
            -detection["confidence"],
        )
    )

    unique_detections = {}

    for detection in all_detections:
        frame = detection["frame"]

        if (
            frame
            not in unique_detections
            or detection["confidence"]
            > unique_detections[
                frame
            ]["confidence"]
        ):
            unique_detections[
                frame
            ] = detection

    merged_detections = list(
        unique_detections.values()
    )

    merged_detections.sort(
        key=lambda detection: (
            detection["frame"]
        )
    )

    class_result = get_stable_class(
        all_classes
    )

    if class_result is None:
        return None

    (
        stable_class,
        votes,
        counts,
    ) = class_result

    class_confidence = (
        votes
        / len(all_classes)
    )

    return {
        "stable_class": stable_class,
        "class_confidence": round(
            class_confidence,
            4,
        ),
        "class_votes": dict(
            counts
        ),
        "detections": (
            merged_detections
        ),
        "rejected_detections": (
            rejected_detections
        ),
        "source_tracker_ids": sorted(
            track_ids
        ),
    }


def build_stitch_chains(
    valid_tracks,
    frame_width,
    frame_height,
    fps,
):
    print(
        "[tracking] Building stitch "
        "candidates..."
    )

    track_ids = sorted(
        valid_tracks.keys(),
        key=lambda track_id: (
            valid_tracks[
                track_id
            ]["detections"][0][
                "frame"
            ]
        ),
    )

    candidates = []

    for track_id_a in track_ids:
        for track_id_b in track_ids:
            if (
                track_id_a
                == track_id_b
            ):
                continue

            track_a = valid_tracks[
                track_id_a
            ]

            track_b = valid_tracks[
                track_id_b
            ]

            if (
                track_a["detections"][-1][
                    "frame"
                ]
                >=
                track_b["detections"][0][
                    "frame"
                ]
            ):
                continue

            result = calculate_stitch_score(
                track_a,
                track_b,
                frame_width,
                frame_height,
                fps,
            )

            if result is None:
                continue

            if (
                result["score"]
                < MIN_STITCH_SCORE
            ):
                continue

            candidates.append(
                (
                    result["score"],
                    track_id_a,
                    track_id_b,
                    result,
                )
            )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    successor = {}

    predecessor = {}

    accepted_stitches = []

    for (
        score,
        track_id_a,
        track_id_b,
        stitch_result,
    ) in candidates:
        if (
            track_id_a
            in successor
        ):
            continue

        if (
            track_id_b
            in predecessor
        ):
            continue

        current = track_id_a

        cycle_found = False

        while current in predecessor:
            current = predecessor[
                current
            ]

            if current == track_id_b:
                cycle_found = True

                break

        if cycle_found:
            continue

        successor[
            track_id_a
        ] = track_id_b

        predecessor[
            track_id_b
        ] = track_id_a

        accepted_stitches.append(
            {
                "from_tracker_id": (
                    track_id_a
                ),
                "to_tracker_id": (
                    track_id_b
                ),
                **stitch_result,
            }
        )

        print(
            f"  stitched Track "
            f"{track_id_a} -> "
            f"Track {track_id_b} "
            f"(score={score:.3f}, "
            f"gap="
            f"{stitch_result['gap_seconds']}"
            f"s)"
        )

    chains = []

    visited = set()

    for track_id in track_ids:
        if (
            track_id
            in predecessor
        ):
            continue

        chain = []

        current = track_id

        while (
            current is not None
            and current not in visited
        ):
            chain.append(current)

            visited.add(current)

            current = successor.get(
                current
            )

        if chain:
            chains.append(chain)

    for track_id in track_ids:
        if track_id in visited:
            continue

        chains.append(
            [track_id]
        )

    return (
        chains,
        accepted_stitches,
    )


def stitch_tracks(
    valid_tracks,
    frame_width,
    frame_height,
    fps,
):
    print(
        "[tracking] Stitching fragmented "
        "tracks..."
    )

    (
        chains,
        accepted_stitches,
    ) = build_stitch_chains(
        valid_tracks,
        frame_width,
        frame_height,
        fps,
    )

    stitched_tracks = {}

    stitched_id = 1

    for chain in chains:
        merged_track = merge_track_group(
            chain,
            valid_tracks,
        )

        if merged_track is None:
            continue

        stitched_tracks[
            stitched_id
        ] = merged_track

        stitched_id += 1

    print(
        f"[tracking] Valid raw tracks: "
        f"{len(valid_tracks)}"
    )

    print(
        f"[tracking] Stitch chains: "
        f"{len(chains)}"
    )

    print(
        f"[tracking] Stitched entities: "
        f"{len(stitched_tracks)}"
    )

    print(
        f"[tracking] Accepted stitches: "
        f"{len(accepted_stitches)}"
    )

    return (
        stitched_tracks,
        accepted_stitches,
    )


def calculate_entity_quality(
    track,
    fps,
):
    detections = track[
        "detections"
    ]

    if not detections:
        return {
            "score": 0.0,
            "duration_score": 0.0,
            "frame_score": 0.0,
            "confidence_score": 0.0,
            "continuity_score": 0.0,
            "size_score": 0.0,
        }

    first_frame = detections[0][
        "frame"
    ]

    last_frame = detections[-1][
        "frame"
    ]

    duration_seconds = (
        last_frame
        - first_frame
    ) / fps

    duration_score = min(
        1.0,
        duration_seconds / 2.0,
    )

    frame_score = min(
        1.0,
        len(detections) / 45.0,
    )

    average_confidence = (
        get_average_confidence(
            detections
        )
    )

    confidence_score = min(
        1.0,
        average_confidence,
    )

    continuity_score = (
        calculate_track_continuity(
            detections
        )
    )

    average_area = (
        get_average_area_ratio(
            detections
        )
    )

    size_score = min(
        1.0,
        average_area / 0.015,
    )

    score = (
        0.22 * duration_score
        + 0.20 * frame_score
        + 0.23 * confidence_score
        + 0.20 * continuity_score
        + 0.15 * size_score
    )

    return {
        "score": round(
            score,
            4,
        ),
        "duration_score": round(
            duration_score,
            4,
        ),
        "frame_score": round(
            frame_score,
            4,
        ),
        "confidence_score": round(
            confidence_score,
            4,
        ),
        "continuity_score": round(
            continuity_score,
            4,
        ),
        "size_score": round(
            size_score,
            4,
        ),
        "average_detection_confidence": round(
            average_confidence,
            4,
        ),
        "average_area_ratio": round(
            average_area,
            6,
        ),
    }


def filter_stitched_entities(
    stitched_tracks,
    fps,
):
    print(
        "[tracking] Validating physical "
        "entities..."
    )

    accepted = {}

    rejected = {}

    new_entity_id = 1

    for (
        stitched_id,
        track,
    ) in stitched_tracks.items():
        detections = track[
            "detections"
        ]

        first_frame = detections[0][
            "frame"
        ]

        last_frame = detections[-1][
            "frame"
        ]

        duration_seconds = (
            last_frame
            - first_frame
        ) / fps

        quality = calculate_entity_quality(
            track,
            fps,
        )

        rejection_reasons = []

        if (
            len(detections)
            < MIN_ENTITY_FRAMES
        ):
            rejection_reasons.append(
                "too_few_frames"
            )

        if (
            duration_seconds
            < MIN_ENTITY_DURATION_SECONDS
        ):
            rejection_reasons.append(
                "short_duration"
            )

        if (
            quality["score"]
            < MIN_ENTITY_QUALITY_SCORE
        ):
            rejection_reasons.append(
                "low_quality"
            )

        track["entity_quality"] = quality

        if rejection_reasons:
            rejected[
                stitched_id
            ] = {
                "stable_class": (
                    track["stable_class"]
                ),
                "source_tracker_ids": (
                    track[
                        "source_tracker_ids"
                    ]
                ),
                "frames_seen": len(
                    detections
                ),
                "duration_seconds": round(
                    duration_seconds,
                    3,
                ),
                "quality": quality,
                "rejection_reasons": (
                    rejection_reasons
                ),
            }

            continue

        accepted[
            new_entity_id
        ] = track

        new_entity_id += 1

    print(
        f"[tracking] Accepted entities: "
        f"{len(accepted)}"
    )

    print(
        f"[tracking] Rejected weak entities: "
        f"{len(rejected)}"
    )

    return (
        accepted,
        rejected,
    )


def build_entity_names(
    valid_tracks,
):
    class_counters = defaultdict(
        int
    )

    entity_names = {}

    ordered_tracks = sorted(
        valid_tracks.items(),
        key=lambda item: (
            item[1][
                "detections"
            ][0]["frame"],
            item[0],
        ),
    )

    for (
        track_id,
        track,
    ) in ordered_tracks:
        stable_class = track[
            "stable_class"
        ]

        class_counters[
            stable_class
        ] += 1

        prefix = (
            stable_class.capitalize()
        )

        entity_names[
            track_id
        ] = (
            f"{prefix}_"
            f"{class_counters[stable_class]}"
        )

    return entity_names


def track_video(
    video_path,
    output_json,
    output_video,
):
    print(
        f"[tracking] Loading "
        f"{MODEL_NAME}..."
    )

    model = YOLO(
        MODEL_NAME
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 30.0

    cap.release()

    print(
        f"[tracking] Video: "
        f"{frame_width}x"
        f"{frame_height}, "
        f"{fps:.2f} FPS, "
        f"{total_frames} frames"
    )

    minimum_track_frames = max(
        MIN_TRACK_FRAMES_ABSOLUTE,
        int(
            fps
            * MIN_TRACK_DURATION_SECONDS
        ),
    )

    print(
        f"[tracking] Minimum raw track "
        f"frames: {minimum_track_frames}"
    )

    print(
        f"[tracking] Maximum stitch gap: "
        f"{MAX_STITCH_GAP_SECONDS:.2f}s "
        f"({int(fps * MAX_STITCH_GAP_SECONDS)} "
        f"frames)"
    )

    os.makedirs(
        os.path.dirname(
            output_json
        ) or ".",
        exist_ok=True,
    )

    os.makedirs(
        os.path.dirname(
            output_video
        ) or ".",
        exist_ok=True,
    )

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (
            frame_width,
            frame_height,
        ),
    )

    tracks = defaultdict(
        lambda: {
            "class_history": [],
            "detections": [],
            "rejected_detections": 0,
        }
    )

    rejected_stats = Counter()

    frame_index = 0

    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        classes=list(
            TRACKED_CLASSES.keys()
        ),
        conf=0.25,
        iou=0.5,
        verbose=False,
    )

    for result in results:
        frame = result.orig_img.copy()

        if (
            result.boxes is None
            or result.boxes.id is None
        ):
            writer.write(frame)

            frame_index += 1

            continue

        boxes = result.boxes

        xyxy_values = (
            boxes.xyxy
            .cpu()
            .tolist()
        )

        class_values = (
            boxes.cls
            .cpu()
            .tolist()
        )

        confidence_values = (
            boxes.conf
            .cpu()
            .tolist()
        )

        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )

        for (
            box,
            class_id,
            confidence,
            track_id,
        ) in zip(
            xyxy_values,
            class_values,
            confidence_values,
            track_ids,
        ):
            class_id = int(
                class_id
            )

            track_id = int(
                track_id
            )

            if (
                class_id
                not in TRACKED_CLASSES
            ):
                continue

            class_name = (
                TRACKED_CLASSES[
                    class_id
                ]
            )

            (
                reject,
                reason,
            ) = reject_detection(
                box,
                frame_width,
                frame_height,
            )

            if reject:
                tracks[track_id][
                    "rejected_detections"
                ] += 1

                rejected_stats[
                    reason
                ] += 1

                continue

            center = box_center(
                box
            )

            detection = {
                "frame": frame_index,
                "time_seconds": round(
                    frame_index / fps,
                    3,
                ),
                "center": center,
                "bbox": round_box(
                    box
                ),
                "confidence": round(
                    float(confidence),
                    4,
                ),
                "detected_class": (
                    class_name
                ),
                "area_ratio": round(
                    get_area_ratio(
                        box,
                        frame_width,
                        frame_height,
                    ),
                    6,
                ),
            }

            tracks[track_id][
                "class_history"
            ].append(
                class_name
            )

            tracks[track_id][
                "detections"
            ].append(
                detection
            )

            x1, y1, x2, y2 = [
                int(value)
                for value in box
            ]

            label = (
                f"Track {track_id} "
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        y1 - 8,
                        20,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

        writer.write(
            frame
        )

        frame_index += 1

        if (
            frame_index % 25
            == 0
        ):
            print(
                f"[tracking] Processed "
                f"{frame_index}/"
                f"{total_frames} frames"
            )

    writer.release()

    print(
        "[tracking] Consolidating "
        "raw tracks..."
    )

    valid_tracks = {}

    raw_track_rejections = {}

    for (
        track_id,
        track,
    ) in tracks.items():
        detections = track[
            "detections"
        ]

        rejection_reasons = []

        if (
            len(detections)
            < minimum_track_frames
        ):
            rejection_reasons.append(
                "too_few_frames"
            )

        average_confidence = (
            get_average_confidence(
                detections
            )
        )

        if (
            average_confidence
            < MIN_AVERAGE_DETECTION_CONFIDENCE
        ):
            rejection_reasons.append(
                "low_detection_confidence"
            )

        if rejection_reasons:
            raw_track_rejections[
                track_id
            ] = {
                "frames_seen": len(
                    detections
                ),
                "average_confidence": round(
                    average_confidence,
                    4,
                ),
                "rejection_reasons": (
                    rejection_reasons
                ),
            }

            continue

        class_result = (
            get_stable_class(
                track[
                    "class_history"
                ]
            )
        )

        if class_result is None:
            continue

        (
            stable_class,
            votes,
            class_counts,
        ) = class_result

        class_confidence = (
            votes
            / len(
                track[
                    "class_history"
                ]
            )
        )

        valid_tracks[
            track_id
        ] = {
            "stable_class": (
                stable_class
            ),
            "class_confidence": round(
                class_confidence,
                4,
            ),
            "class_votes": dict(
                class_counts
            ),
            "detections": detections,
            "rejected_detections": (
                track[
                    "rejected_detections"
                ]
            ),
        }

    (
        stitched_tracks,
        accepted_stitches,
    ) = stitch_tracks(
        valid_tracks,
        frame_width,
        frame_height,
        fps,
    )

    (
        physical_entities,
        rejected_entities,
    ) = filter_stitched_entities(
        stitched_tracks,
        fps,
    )

    entity_names = build_entity_names(
        physical_entities
    )

    object_counts = {
        "cars": 0,
        "buses": 0,
        "motorcycles": 0,
        "bicycles": 0,
        "trucks": 0,
        "people": 0,
    }

    tracked_entities = []

    trajectories = {}

    for (
        stitched_id,
        track,
    ) in physical_entities.items():
        stable_class = track[
            "stable_class"
        ]

        entity_id = entity_names[
            stitched_id
        ]

        count_key = COUNT_KEYS[
            stable_class
        ]

        object_counts[
            count_key
        ] += 1

        detections = track[
            "detections"
        ]

        first_frame = detections[0][
            "frame"
        ]

        last_frame = detections[-1][
            "frame"
        ]

        tracked_entities.append(
            {
                "id": entity_id,
                "type": stable_class,
                "source_tracker_ids": (
                    track[
                        "source_tracker_ids"
                    ]
                ),
                "frames_seen": len(
                    detections
                ),
                "first_frame": (
                    first_frame
                ),
                "last_frame": (
                    last_frame
                ),
                "duration_seconds": round(
                    (
                        last_frame
                        - first_frame
                    )
                    / fps,
                    3,
                ),
                "class_confidence": (
                    track[
                        "class_confidence"
                    ]
                ),
                "class_votes": (
                    track[
                        "class_votes"
                    ]
                ),
                "entity_quality": (
                    track[
                        "entity_quality"
                    ]
                ),
                "rejected_detections": (
                    track[
                        "rejected_detections"
                    ]
                ),
            }
        )

        trajectories[
            entity_id
        ] = detections

    tracked_entities.sort(
        key=lambda entity: (
            entity["first_frame"],
            entity[
                "source_tracker_ids"
            ][0],
        )
    )

    output = {
        "video_metadata": {
            "video_path": video_path,
            "width": frame_width,
            "height": frame_height,
            "fps": round(
                fps,
                3,
            ),
            "total_frames": (
                total_frames
            ),
            "duration_seconds": round(
                total_frames / fps,
                3,
            ),
        },
        "tracking_config": {
            "model": MODEL_NAME,
            "tracker": "ByteTrack",
            "minimum_track_duration_seconds": (
                MIN_TRACK_DURATION_SECONDS
            ),
            "minimum_raw_track_frames": (
                minimum_track_frames
            ),
            "minimum_average_detection_confidence": (
                MIN_AVERAGE_DETECTION_CONFIDENCE
            ),
            "minimum_entity_duration_seconds": (
                MIN_ENTITY_DURATION_SECONDS
            ),
            "minimum_entity_frames": (
                MIN_ENTITY_FRAMES
            ),
            "minimum_entity_quality_score": (
                MIN_ENTITY_QUALITY_SCORE
            ),
            "maximum_box_area_ratio": (
                MAX_BOX_AREA_RATIO
            ),
            "dashcam_area_ratio": (
                DASHCAM_AREA_RATIO
            ),
            "dashcam_y_threshold": (
                DASHCAM_Y_THRESHOLD
            ),
            "track_stitching": {
                "enabled": True,
                "maximum_gap_seconds": (
                    MAX_STITCH_GAP_SECONDS
                ),
                "maximum_distance_ratio": (
                    MAX_STITCH_DISTANCE_RATIO
                ),
                "maximum_area_change_ratio": (
                    MAX_AREA_CHANGE_RATIO
                ),
                "minimum_direction_similarity": (
                    MIN_DIRECTION_SIMILARITY
                ),
                "minimum_stitch_score": (
                    MIN_STITCH_SCORE
                ),
            },
        },
        "object_counts": (
            object_counts
        ),
        "tracking_summary": {
            "raw_tracker_ids": len(
                tracks
            ),
            "valid_raw_tracks": len(
                valid_tracks
            ),
            "raw_tracks_rejected": len(
                raw_track_rejections
            ),
            "stitched_candidates": len(
                stitched_tracks
            ),
            "physical_entities": len(
                physical_entities
            ),
            "weak_entities_rejected": len(
                rejected_entities
            ),
            "accepted_stitches": len(
                accepted_stitches
            ),
            "rejected_detections": dict(
                rejected_stats
            ),
        },
        "raw_track_rejections": (
            raw_track_rejections
        ),
        "track_stitches": (
            accepted_stitches
        ),
        "rejected_entities": (
            rejected_entities
        ),
        "tracked_entities": (
            tracked_entities
        ),
        "trajectories": (
            trajectories
        ),
    }

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

    print(
        "=" * 72
    )

    print(
        "TRACKING COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "\nPhysical object counts:"
    )

    for (
        key,
        value,
    ) in object_counts.items():
        print(
            f"  {key:<15} "
            f"{value}"
        )

    print(
        "\nTracking summary:"
    )

    print(
        f"  Raw tracker IDs: "
        f"{len(tracks)}"
    )

    print(
        f"  Valid raw tracks: "
        f"{len(valid_tracks)}"
    )

    print(
        f"  Raw tracks rejected: "
        f"{len(raw_track_rejections)}"
    )

    print(
        f"  Stitched candidates: "
        f"{len(stitched_tracks)}"
    )

    print(
        f"  Accepted stitches: "
        f"{len(accepted_stitches)}"
    )

    print(
        f"  Physical entities: "
        f"{len(physical_entities)}"
    )

    print(
        f"  Weak entities rejected: "
        f"{len(rejected_entities)}"
    )

    print(
        f"  Giant boxes rejected: "
        f"{rejected_stats['giant_box']}"
    )

    print(
        f"  Dashcam detections rejected: "
        f"{rejected_stats['dashcam_foreground']}"
    )

    print(
        f"\nJSON written to: "
        f"{output_json}"
    )

    print(
        f"Annotated video written to: "
        f"{output_video}"
    )

    print(
        "=" * 72
    )

    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "YOLO + ByteTrack traffic "
            "tracker with physical entity "
            "consolidation"
        )
    )

    parser.add_argument(
        "video",
        help="Path to input video",
    )

    parser.add_argument(
        "--output_json",
        default=None,
        help="Output tracking JSON",
    )

    parser.add_argument(
        "--output_video",
        default=None,
        help="Output annotated video",
    )

    args = parser.parse_args()

    video_name = os.path.splitext(
        os.path.basename(
            args.video
        )
    )[0]

    output_json = (
        args.output_json
        or (
            f"results/"
            f"{video_name}_tracking.json"
        )
    )

    output_video = (
        args.output_video
        or (
            f"results/"
            f"{video_name}_tracked.mp4"
        )
    )

    track_video(
        args.video,
        output_json,
        output_video,
    )


if __name__ == "__main__":
    main()
