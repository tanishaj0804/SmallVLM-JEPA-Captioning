from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

TOP_TEMPORAL_WINDOWS = 5
TEMPORAL_SUPPORT_THRESHOLD = 0.70
INTERACTION_TRIGGER_THRESHOLD = 0.35
ANCHOR_MERGE_SECONDS = 0.85
TEMPORAL_ATTACH_SECONDS = 1.25
EVENT_CONTEXT_BEFORE = 0.75
EVENT_CONTEXT_AFTER = 1.25

SIGNAL_NAMES = [
    "motion_change",
    "proximity_change",
    "track_disruption",
    "geometry_change",
    "local_motion_chaos",
    "motion_event_density",
]
DISRUPTIVE_SIGNALS = {
    "track_disruption",
    "local_motion_chaos",
    "motion_event_density",
}
SIGNATURE_ACTIVE_THRESHOLD = 0.60

CONSEQUENCE_SIGNALS = {
    "motion_change",
    "track_disruption",
    "geometry_change",
}
INTERACTION_SIGNALS = {
    "proximity_change",
}
SCENE_ACTIVITY_SIGNALS = {
    "local_motion_chaos",
    "motion_event_density",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def sf(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_view_spec(spec: str) -> Tuple[str, Path, Path]:
    view, raw = spec.split("=", 1)
    interaction, temporal = raw.split(",", 1)
    return view.strip().upper(), Path(interaction), Path(temporal)


def extract_interaction_anchors(view: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    anchors = []
    for episode in data.get("interaction_episodes", []):
        chain = episode.get("event_chain", {})
        trigger = chain.get("trigger", {})
        response = chain.get("response", {})
        trigger_score = sf(trigger.get("score"))
        response_events = response.get("response_events", [])
        chain_present = bool(chain.get("temporal_chain_present"))
        current_critical = bool(episode.get("critical_event_candidate"))

        if (
            trigger_score < INTERACTION_TRIGGER_THRESHOLD
            and not chain_present
            and not current_critical
        ):
            continue

        trigger_time = sf(
            trigger.get("time_seconds"),
            sf(episode.get("peak_interaction_time")),
        )

        anchors.append({
            "view": view,
            "anchor_type": "interaction_trigger",
            "anchor_time": trigger_time,
            "interaction_episode_id": episode.get("interaction_episode_id"),
            "entity_a": episode.get("entity_a"),
            "entity_b": episode.get("entity_b"),
            "class_a": episode.get("class_a"),
            "class_b": episode.get("class_b"),
            "trigger_type": trigger.get("trigger_type"),
            "trigger_score": trigger_score,
            "peak_interaction_score": sf(episode.get("peak_interaction_score")),
            "response_score": sf(response.get("score")),
            "strong_response_score": sf(response.get("strong_response_score")),
            "response_events": response_events,
            "post_trigger_separation": chain.get("post_trigger_separation", {}),
            "temporal_chain_present": chain_present,
            "current_critical_candidate": current_critical,
            "critical_event_score": sf(episode.get("critical_event_score")),
        })
    return anchors


def extract_temporal_windows(view: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []
    windows = data.get("ranked_windows", data.get("top_candidates", []))
    for window in windows[:TOP_TEMPORAL_WINDOWS]:
        score = sf(window.get("abnormality_score"))
        if score < TEMPORAL_SUPPORT_THRESHOLD:
            continue
        start = sf(window.get("start_time"))
        end = sf(window.get("end_time"))
        normalized = {
            name: sf(window.get("normalized_signals", {}).get(name))
            for name in SIGNAL_NAMES
        }
        result.append({
            "view": view,
            "window_id": window.get("window_id"),
            "start_time": start,
            "end_time": end,
            "center_time": 0.5 * (start + end),
            "abnormality_score": score,
            "state_score": sf(window.get("state_score")),
            "transition_score": sf(window.get("transition_score")),
            "normalized_signals": normalized,
            "raw_signals": window.get("raw_signals", {}),
            "evidence": window.get("evidence", {}),
        })
    return result


def cluster_anchor_times(anchors: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    anchors = sorted(anchors, key=lambda x: x["anchor_time"])
    clusters: List[List[Dict[str, Any]]] = []
    for anchor in anchors:
        best_index = None
        best_distance = None
        for index, cluster in enumerate(clusters):
            center = sum(x["anchor_time"] for x in cluster) / len(cluster)
            distance = abs(anchor["anchor_time"] - center)
            if distance <= ANCHOR_MERGE_SECONDS and (
                best_distance is None or distance < best_distance
            ):
                best_index = index
                best_distance = distance
        if best_index is None:
            clusters.append([anchor])
        else:
            clusters[best_index].append(anchor)
    return clusters


def temporal_only_anchors(
    temporal_windows: List[Dict[str, Any]],
    anchor_clusters: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    result = []
    existing_centers = [
        sum(x["anchor_time"] for x in cluster) / len(cluster)
        for cluster in anchor_clusters
    ]
    for window in temporal_windows:
        center = window["center_time"]
        if any(abs(center - t) <= TEMPORAL_ATTACH_SECONDS for t in existing_centers):
            continue
        result.append({
            "view": window["view"],
            "anchor_type": "temporal_only",
            "anchor_time": center,
            "trigger_type": None,
            "trigger_score": 0.0,
            "peak_interaction_score": 0.0,
            "response_score": 0.0,
            "strong_response_score": 0.0,
            "response_events": [],
            "post_trigger_separation": {},
            "temporal_chain_present": False,
            "current_critical_candidate": False,
            "critical_event_score": 0.0,
        })
    return result


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    va = [sf(a.get(name)) for name in SIGNAL_NAMES]
    vb = [sf(b.get(name)) for name in SIGNAL_NAMES]
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(va, vb)) / (na * nb)


def summarize_signal_coherence(
    attached: List[Dict[str, Any]],
    all_views: List[str],
) -> Dict[str, Any]:
    best_by_view = {}
    for view in all_views:
        candidates = [w for w in attached if w["view"] == view]
        if candidates:
            best_by_view[view] = max(
                candidates, key=lambda w: w["abnormality_score"]
            )

    pair_scores = []
    pair_details = []
    views = sorted(best_by_view)
    for i, first in enumerate(views):
        for second in views[i + 1:]:
            score = cosine_similarity(
                best_by_view[first]["normalized_signals"],
                best_by_view[second]["normalized_signals"],
            )
            pair_scores.append(score)
            pair_details.append({
                "views": [first, second],
                "cosine_similarity": round(score, 4),
            })

    coherence = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

    active_by_view = {}
    disruptive_by_view = {}
    for view, window in best_by_view.items():
        sig = window["normalized_signals"]
        active_by_view[view] = [
            name for name in SIGNAL_NAMES
            if sf(sig.get(name)) >= SIGNATURE_ACTIVE_THRESHOLD
        ]
        disruptive_by_view[view] = [
            name for name in DISRUPTIVE_SIGNALS
            if sf(sig.get(name)) >= SIGNATURE_ACTIVE_THRESHOLD
        ]

    disruptive_counts = {
        name: sum(name in values for values in disruptive_by_view.values())
        for name in sorted(DISRUPTIVE_SIGNALS)
    }
    shared_disruptive = [
        name for name, count in disruptive_counts.items()
        if count >= 2
    ]

    disruptive_views = sorted([
        view for view, values in disruptive_by_view.items() if values
    ])

    consequence_signals_by_view = {}
    interaction_signals_by_view = {}
    scene_activity_signals_by_view = {}

    for view, window in best_by_view.items():
        sig = window["normalized_signals"]

        consequence_signals_by_view[view] = [
            name for name in sorted(CONSEQUENCE_SIGNALS)
            if sf(sig.get(name)) >= SIGNATURE_ACTIVE_THRESHOLD
        ]
        interaction_signals_by_view[view] = [
            name for name in sorted(INTERACTION_SIGNALS)
            if sf(sig.get(name)) >= SIGNATURE_ACTIVE_THRESHOLD
        ]
        scene_activity_signals_by_view[view] = [
            name for name in sorted(SCENE_ACTIVITY_SIGNALS)
            if sf(sig.get(name)) >= SIGNATURE_ACTIVE_THRESHOLD
        ]

    consequence_views = sorted([
        view for view, values in consequence_signals_by_view.items()
        if values
    ])
    interaction_signal_views = sorted([
        view for view, values in interaction_signals_by_view.items()
        if values
    ])
    scene_activity_views = sorted([
        view for view, values in scene_activity_signals_by_view.items()
        if values
    ])

    consequence_signal_types = sorted({
        signal
        for values in consequence_signals_by_view.values()
        for signal in values
    })

    consequence_observations = sum(
        len(values) for values in consequence_signals_by_view.values()
    )
    consequence_view_ratio = len(consequence_views) / max(1, len(all_views))
    consequence_type_ratio = (
        len(consequence_signal_types) / max(1, len(CONSEQUENCE_SIGNALS))
    )
    consequence_diversity_score = (
        0.65 * consequence_view_ratio
        + 0.35 * consequence_type_ratio
    )

    track_disruption_views = sorted([
        view
        for view, values in consequence_signals_by_view.items()
        if "track_disruption" in values
    ])

    scene_only_views = sorted([
        view for view in best_by_view
        if scene_activity_signals_by_view.get(view)
        and not consequence_signals_by_view.get(view)
        and not interaction_signals_by_view.get(view)
    ])

    # Cross-view semantic role pattern:
    # some cameras may directly observe entity consequence while other cameras
    # only observe the scene response caused by the same synchronized event.
    direct_consequence_views = consequence_views
    indirect_response_views = sorted([
        view for view in scene_activity_views
        if view not in direct_consequence_views
    ])
    complementary_response_pattern = (
        len(direct_consequence_views) >= 1
        and len(indirect_response_views) >= 1
        and len(best_by_view) >= 2
    )

    # Pure replicated kinematic change is especially vulnerable to parallax,
    # turning and camera projection. It is not physical-consequence evidence.
    replicated_kinematic_only = (
        len(consequence_views) >= 2
        and set(consequence_signal_types).issubset(
            {"motion_change", "geometry_change"}
        )
        and len(track_disruption_views) == 0
    )

    return {
        "cross_view_signal_coherence": round(coherence, 4),
        "pairwise_signal_coherence": pair_details,
        "per_view_signal_signatures": {
            view: {
                name: round(sf(window["normalized_signals"].get(name)), 4)
                for name in SIGNAL_NAMES
            }
            for view, window in best_by_view.items()
        },
        "active_signals_by_view": active_by_view,
        "disruptive_signals_by_view": disruptive_by_view,
        "disruptive_views": disruptive_views,
        "shared_disruptive_signals": shared_disruptive,
        "shared_disruptive_signal_count": len(shared_disruptive),
        "consequence_signals_by_view": consequence_signals_by_view,
        "interaction_signals_by_view": interaction_signals_by_view,
        "scene_activity_signals_by_view": scene_activity_signals_by_view,
        "consequence_views": consequence_views,
        "interaction_signal_views": interaction_signal_views,
        "scene_activity_views": scene_activity_views,
        "scene_only_views": scene_only_views,
        "consequence_signal_types": consequence_signal_types,
        "consequence_signal_type_count": len(consequence_signal_types),
        "consequence_observations": consequence_observations,
        "consequence_view_ratio": round(consequence_view_ratio, 4),
        "consequence_diversity_score": round(consequence_diversity_score, 4),
        "track_disruption_views": track_disruption_views,
        "direct_consequence_views": direct_consequence_views,
        "indirect_response_views": indirect_response_views,
        "complementary_response_pattern": complementary_response_pattern,
        "replicated_kinematic_only": replicated_kinematic_only,
    }


def summarize_event(
    index: int,
    cluster: List[Dict[str, Any]],
    temporal_windows: List[Dict[str, Any]],
    all_views: List[str],
) -> Dict[str, Any]:
    anchor_time = sum(x["anchor_time"] for x in cluster) / len(cluster)
    attached = [
        w for w in temporal_windows
        if abs(w["center_time"] - anchor_time) <= TEMPORAL_ATTACH_SECONDS
    ]

    interaction_views = sorted({
        x["view"] for x in cluster
        if x["anchor_type"] == "interaction_trigger"
    })
    temporal_views = sorted({w["view"] for w in attached})
    supporting_views = sorted(set(interaction_views) | set(temporal_views))

    temporal_scores = {
        view: max(
            [w["abnormality_score"] for w in attached if w["view"] == view],
            default=0.0,
        )
        for view in all_views
    }

    signal_analysis = summarize_signal_coherence(attached, all_views)

    max_response = max(
        [max(x["response_score"], x["strong_response_score"]) for x in cluster],
        default=0.0,
    )
    max_trigger = max([x["trigger_score"] for x in cluster], default=0.0)
    max_interaction = max(
        [x["peak_interaction_score"] for x in cluster],
        default=0.0,
    )

    start = max(0.0, anchor_time - EVENT_CONTEXT_BEFORE)
    end = anchor_time + EVENT_CONTEXT_AFTER

    return {
        "fused_event_id": f"FusedEvent_{index}",
        "anchor_time": round(anchor_time, 4),
        "start_time": round(start, 4),
        "end_time": round(end, 4),
        "supporting_views": supporting_views,
        "interaction_views": interaction_views,
        "temporal_views": temporal_views,
        "total_views": len(all_views),
        "temporal_consensus": round(
            len(temporal_views) / max(1, len(all_views)), 4
        ),
        "interaction_consensus": round(
            len(interaction_views) / max(1, len(all_views)), 4
        ),
        "max_temporal_score": round(max(temporal_scores.values(), default=0.0), 4),
        "max_trigger_score": round(max_trigger, 4),
        "max_interaction_score": round(max_interaction, 4),
        "max_response_score": round(max_response, 4),
        "per_view_temporal_scores": {
            k: round(v, 4) for k, v in temporal_scores.items()
        },
        **signal_analysis,
        "interaction_anchors": cluster,
        "attached_temporal_windows": attached,
        "fusion_decision": "LOCALIZED_EVENT_HYPOTHESIS",
        "final_criticality_decision": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--view",
        action="append",
        required=True,
        help="VIEW=interaction_analysis.json,temporal_analysis.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    all_views = []
    all_anchors = []
    all_windows = []
    inputs = {}

    for spec in args.view:
        view, interaction_path, temporal_path = parse_view_spec(spec)
        if not interaction_path.is_file():
            raise FileNotFoundError(interaction_path)
        if not temporal_path.is_file():
            raise FileNotFoundError(temporal_path)

        interaction = load_json(interaction_path)
        temporal = load_json(temporal_path)
        anchors = extract_interaction_anchors(view, interaction)
        windows = extract_temporal_windows(view, temporal)

        all_views.append(view)
        all_anchors.extend(anchors)
        all_windows.extend(windows)
        inputs[view] = {
            "interaction_json": str(interaction_path),
            "temporal_json": str(temporal_path),
        }

        print(
            f"[fusion-v6] {view:<8} interaction anchors={len(anchors):<3} "
            f"temporal windows={len(windows)}"
        )

    clusters = cluster_anchor_times(all_anchors)
    temporal_anchors = temporal_only_anchors(all_windows, clusters)
    if temporal_anchors:
        clusters.extend(cluster_anchor_times(temporal_anchors))

    events = [
        summarize_event(i, cluster, all_windows, all_views)
        for i, cluster in enumerate(clusters, start=1)
    ]
    events.sort(key=lambda x: x["anchor_time"])
    for i, event in enumerate(events, start=1):
        event["fused_event_id"] = f"FusedEvent_{i}"

    output = {
        "configuration": {
            "version": "v6_cross_view_role_fusion",
            "bbox_geometry_fused_across_views": False,
            "cross_camera_reid_required": False,
            "final_accident_classification": False,
            "signature_active_threshold": SIGNATURE_ACTIVE_THRESHOLD,
            "consequence_signals": sorted(CONSEQUENCE_SIGNALS),
            "interaction_signals": sorted(INTERACTION_SIGNALS),
            "scene_activity_signals": sorted(SCENE_ACTIVITY_SIGNALS),
            "signal_names": SIGNAL_NAMES,
            "disruptive_signals": sorted(DISRUPTIVE_SIGNALS),
            "design_note": (
                "Fusion localizes synchronized hypotheses, separates direct entity consequence "
                "from indirect scene response, and identifies complementary cross-view "
                "role patterns. Replicated motion/geometry change without continuity "
                "disruption is retained as projection-vulnerable kinematic evidence."
            ),
        },
        "case": args.case,
        "views": all_views,
        "view_inputs": inputs,
        "fused_events": events,
        "summary": {
            "interaction_anchors": len(all_anchors),
            "temporal_windows": len(all_windows),
            "localized_events": len(events),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 96)
    print("MULTIVIEW EVENT FUSION V6 COMPLETE")
    print("=" * 96)
    print(f"Case             : {args.case}")
    print(f"Views            : {', '.join(all_views)}")
    print(f"Localized events : {len(events)}")
    print(f"Saved to         : {args.output}")
    print("=" * 96)

    for event in events:
        print(
            f"{event['fused_event_id']:<14} | "
            f"anchor={event['anchor_time']:.2f}s | "
            f"window={event['start_time']:.2f}-{event['end_time']:.2f}s | "
            f"temp={event['temporal_consensus']:.2f} | "
            f"inter={event['interaction_consensus']:.2f} | "
            f"coh={event['cross_view_signal_coherence']:.3f} | "
            f"cons_views={event['consequence_views']} | "
            f"cons_types={event['consequence_signal_types']} | "
            f"cons_div={event['consequence_diversity_score']:.3f} | "
            f"direct={event['direct_consequence_views']} | "
            f"indirect={event['indirect_response_views']} | "
            f"complement={event['complementary_response_pattern']} | "
            f"kin_only={event['replicated_kinematic_only']} | "
            f"views={event['supporting_views']}"
        )


if __name__ == "__main__":
    main()
