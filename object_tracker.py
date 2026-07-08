"""
Traffic object detection and tracking using YOLO + ByteTrack.
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

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

MIN_TRACK_FRAMES = 5

MAX_BOX_AREA_RATIO = 0.35
DASHCAM_AREA_RATIO = 0.20
DASHCAM_Y_THRESHOLD = 0.60

MAX_STITCH_FRAME_GAP = 15
MAX_STITCH_DISTANCE_RATIO = 0.15
MAX_AREA_CHANGE_RATIO = 3.0
MIN_DIRECTION_SIMILARITY = -0.20


def round_box(box):
    return [
        round(float(value), 2)
        for value in box
    ]


def box_center(box):
    x1, y1, x2, y2 = box

    return [
        round((x1 + x2) / 2, 2),
        round((y1 + y2) / 2, 2),
    ]


def get_area_ratio(
    box,
    frame_width,
    frame_height,
):
    x1, y1, x2, y2 = box

    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)

    box_area = box_width * box_height
    frame_area = frame_width * frame_height

    if frame_area == 0:
        return 0.0

    return box_area / frame_area


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

    if area_ratio > MAX_BOX_AREA_RATIO:
        return True, "giant_box"

    if (
        y1 > frame_height * DASHCAM_Y_THRESHOLD
        and area_ratio > DASHCAM_AREA_RATIO
    ):
        return True, "dashcam_foreground"

    return False, None


def get_stable_class(class_history):
    if not class_history:
        return None

    counts = Counter(class_history)

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
        class_a in TWO_WHEELER_CLASSES
        and class_b in TWO_WHEELER_CLASSES
    ):
        return True

    return False


def center_distance(
    center_a,
    center_b,
):
    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]

    return math.sqrt(
        dx * dx + dy * dy
    )


def get_motion_vector(
    detections,
    use_end=True,
    window=5,
):
    if len(detections) < 2:
        return None

    if use_end:
        selected = detections[-window:]
    else:
        selected = detections[:window]

    if len(selected) < 2:
        return None

    first = selected[0]["center"]
    last = selected[-1]["center"]

    return (
        last[0] - first[0],
        last[1] - first[1],
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
        ax * ax + ay * ay
    )

    magnitude_b = math.sqrt(
        bx * bx + by * by
    )

    if (
        magnitude_a < 1e-6
        or magnitude_b < 1e-6
    ):
        return None

    return (
        ax * bx + ay * by
    ) / (
        magnitude_a * magnitude_b
    )


def get_area_change_ratio(
    detection_a,
    detection_b,
):
    area_a = detection_a["area_ratio"]
    area_b = detection_b["area_ratio"]

    if area_a <= 0 or area_b <= 0:
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

    last_detection = detections[-1]

    if len(detections) < 2:
        return last_detection["center"]

    selected = detections[-5:]

    first = selected[0]
    last = selected[-1]

    frame_difference = (
        last["frame"] - first["frame"]
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
        target_frame - last["frame"]
    )

    return [
        last["center"][0]
        + velocity_x * future_frames,
        last["center"][1]
        + velocity_y * future_frames,
    ]


def calculate_stitch_score(
    track_a,
    track_b,
    frame_width,
    frame_height,
):
    detections_a = track_a["detections"]
    detections_b = track_b["detections"]

    last_a = detections_a[-1]
    first_b = detections_b[0]

    frame_gap = (
        first_b["frame"]
        - last_a["frame"]
    )

    if frame_gap <= 0:
        return None

    if frame_gap > MAX_STITCH_FRAME_GAP:
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
        direction_similarity is not None
        and direction_similarity
        < MIN_DIRECTION_SIMILARITY
    ):
        return None

    gap_score = 1.0 - (
        frame_gap
        / MAX_STITCH_FRAME_GAP
    )

    distance_score = 1.0 - (
        distance_ratio
        / MAX_STITCH_DISTANCE_RATIO
    )

    area_score = 1.0 / (
        area_change_ratio
    )

    if direction_similarity is None:
        direction_score = 0.5
    else:
        direction_score = (
            direction_similarity + 1.0
        ) / 2.0

    class_score = (
        1.0
        if (
            track_a["stable_class"]
            == track_b["stable_class"]
        )
        else 0.6
    )

    score = (
        0.20 * gap_score
        + 0.35 * distance_score
        + 0.15 * area_score
        + 0.20 * direction_score
        + 0.10 * class_score
    )

    return {
        "score": round(score, 4),
        "frame_gap": frame_gap,
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
            if direction_similarity is None
            else round(
                direction_similarity,
                4,
            )
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
        track = valid_tracks[track_id]

        all_detections.extend(
            track["detections"]
        )

        for class_name, count in (
            track["class_votes"].items()
        ):
            all_classes.extend(
                [class_name] * count
            )

        rejected_detections += track[
            "rejected_detections"
        ]

    all_detections.sort(
        key=lambda detection: (
            detection["frame"],
            detection["confidence"],
        )
    )

    unique_detections = {}

    for detection in all_detections:
        frame = detection["frame"]

        if (
            frame not in unique_detections
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

    stable_class, votes, counts = (
        class_result
    )

    class_confidence = (
        votes / len(all_classes)
    )

    return {
        "stable_class": stable_class,
        "class_confidence": round(
            class_confidence,
            4,
        ),
        "class_votes": dict(counts),
        "detections": merged_detections,
        "rejected_detections": (
            rejected_detections
        ),
        "source_tracker_ids": sorted(
            track_ids
        ),
    }


def stitch_tracks(
    valid_tracks,
    frame_width,
    frame_height,
):
    print(
        "[tracking] Stitching fragmented "
        "tracks..."
    )

    track_ids = sorted(
        valid_tracks.keys(),
        key=lambda track_id: (
            valid_tracks[
                track_id
            ]["detections"][0]["frame"]
        ),
    )

    parent = {
        track_id: track_id
        for track_id in track_ids
    }

    def find(track_id):
        while parent[track_id] != track_id:
            parent[track_id] = parent[
                parent[track_id]
            ]

            track_id = parent[track_id]

        return track_id

    def union(track_a, track_b):
        root_a = find(track_a)
        root_b = find(track_b)

        if root_a != root_b:
            parent[root_b] = root_a

    candidates = []

    for track_id_a in track_ids:
        for track_id_b in track_ids:
            if track_id_a == track_id_b:
                continue

            track_a = valid_tracks[
                track_id_a
            ]

            track_b = valid_tracks[
                track_id_b
            ]

            stitch_result = (
                calculate_stitch_score(
                    track_a,
                    track_b,
                    frame_width,
                    frame_height,
                )
            )

            if stitch_result is None:
                continue

            candidates.append(
                (
                    stitch_result["score"],
                    track_id_a,
                    track_id_b,
                    stitch_result,
                )
            )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    used_as_previous = set()
    used_as_next = set()

    accepted_stitches = []

    MIN_STITCH_SCORE = 0.62

    for (
        score,
        track_id_a,
        track_id_b,
        stitch_result,
    ) in candidates:
        if score < MIN_STITCH_SCORE:
            continue

        if track_id_a in used_as_previous:
            continue

        if track_id_b in used_as_next:
            continue

        if find(track_id_a) == find(
            track_id_b
        ):
            continue

        union(
            track_id_a,
            track_id_b,
        )

        used_as_previous.add(track_id_a)
        used_as_next.add(track_id_b)

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
            f"(score={score:.3f})"
        )

    groups = defaultdict(list)

    for track_id in track_ids:
        groups[find(track_id)].append(
            track_id
        )

    stitched_tracks = {}

    stitched_id = 1

    for track_group in groups.values():
        merged_track = merge_track_group(
            track_group,
            valid_tracks,
        )

        stitched_tracks[
            stitched_id
        ] = merged_track

        stitched_id += 1

    print(
        f"[tracking] Valid raw tracks: "
        f"{len(valid_tracks)}"
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


def build_entity_names(valid_tracks):
    class_counters = defaultdict(int)
    entity_names = {}

    ordered_tracks = sorted(
        valid_tracks.items(),
        key=lambda item: (
            item[1]["detections"][0]["frame"],
            item[0],
        ),
    )

    for track_id, track in ordered_tracks:
        stable_class = track["stable_class"]

        class_counters[
            stable_class
        ] += 1

        prefix = stable_class.capitalize()

        entity_names[track_id] = (
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

    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(video_path)

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
        f"{frame_width}x{frame_height}, "
        f"{fps:.2f} FPS, "
        f"{total_frames} frames"
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
            boxes.xyxy.cpu().tolist()
        )

        class_values = (
            boxes.cls.cpu().tolist()
        )

        confidence_values = (
            boxes.conf.cpu().tolist()
        )

        track_ids = (
            boxes.id.int().cpu().tolist()
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
            class_id = int(class_id)
            track_id = int(track_id)

            if (
                class_id
                not in TRACKED_CLASSES
            ):
                continue

            class_name = (
                TRACKED_CLASSES[class_id]
            )

            reject, reason = (
                reject_detection(
                    box,
                    frame_width,
                    frame_height,
                )
            )

            if reject:
                tracks[track_id][
                    "rejected_detections"
                ] += 1

                rejected_stats[
                    reason
                ] += 1

                continue

            center = box_center(box)

            detection = {
                "frame": frame_index,
                "time_seconds": round(
                    frame_index / fps,
                    3,
                ),
                "center": center,
                "bbox": round_box(box),
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
                    4,
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
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(y1 - 8, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        writer.write(frame)

        frame_index += 1

        if frame_index % 25 == 0:
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

    for track_id, track in tracks.items():
        detections = track["detections"]

        if (
            len(detections)
            < MIN_TRACK_FRAMES
        ):
            continue

        class_result = (
            get_stable_class(
                track["class_history"]
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
                track["class_history"]
            )
        )

        valid_tracks[track_id] = {
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
    )

    entity_names = build_entity_names(
        stitched_tracks
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
    ) in stitched_tracks.items():
        stable_class = (
            track["stable_class"]
        )

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

        first_frame = (
            detections[0]["frame"]
        )

        last_frame = (
            detections[-1]["frame"]
        )

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
            "fps": round(fps, 3),
            "total_frames": total_frames,
            "duration_seconds": round(
                total_frames / fps,
                3,
            ),
        },
        "tracking_config": {
            "model": MODEL_NAME,
            "tracker": "ByteTrack",
            "minimum_track_frames": (
                MIN_TRACK_FRAMES
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
                "maximum_frame_gap": (
                    MAX_STITCH_FRAME_GAP
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
                    0.62
                ),
            },
        },
        "object_counts": object_counts,
        "tracking_summary": {
            "raw_tracker_ids": len(
                tracks
            ),
            "valid_raw_tracks": len(
                valid_tracks
            ),
            "stitched_entities": len(
                stitched_tracks
            ),
            "filtered_short_tracks": (
                len(tracks)
                - len(valid_tracks)
            ),
            "accepted_stitches": len(
                accepted_stitches
            ),
            "rejected_detections": dict(
                rejected_stats
            ),
        },
        "track_stitches": (
            accepted_stitches
        ),
        "tracked_entities": (
            tracked_entities
        ),
        "trajectories": trajectories,
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

    print("=" * 72)
    print("TRACKING COMPLETE")
    print("=" * 72)

    print(
        "\nStitched object counts:"
    )

    for key, value in (
        object_counts.items()
    ):
        print(
            f"  {key:<15} {value}"
        )

    print("\nTracking summary:")

    print(
        f"  Raw tracker IDs: "
        f"{len(tracks)}"
    )

    print(
        f"  Valid raw tracks: "
        f"{len(valid_tracks)}"
    )

    print(
        f"  Stitched entities: "
        f"{len(stitched_tracks)}"
    )

    print(
        f"  Accepted stitches: "
        f"{len(accepted_stitches)}"
    )

    print(
        f"  Filtered tracks: "
        f"{len(tracks) - len(valid_tracks)}"
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

    print("=" * 72)

    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "YOLO + ByteTrack traffic "
            "tracker with track stitching"
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
