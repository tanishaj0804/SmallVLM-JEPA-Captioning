from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

MIN_CONSEQUENCE_VIEWS = 2
MIN_CONSEQUENCE_DIVERSITY = 0.55
MIN_TEMPORAL_CONSENSUS = 0.67
MIN_TEMPORAL_SCORE = 0.80

RESPONSE_EVENT_TYPES = {
    "rapid_to_low_motion",
    "direction_change",
    "lateral_motion_onset",
    "low_to_rapid_motion",
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


def normalize_response_event(item: Any) -> Dict[str, Any] | None:
    if isinstance(item, str):
        event_type = item
        return {"event_type": event_type, "time_seconds": None}
    if not isinstance(item, dict):
        return None
    event_type = (
        item.get("event_type")
        or item.get("type")
        or item.get("motion_event")
        or item.get("name")
    )
    if not event_type:
        return None
    time_value = (
        item.get("time_seconds")
        if item.get("time_seconds") is not None
        else item.get("timestamp")
    )
    return {
        "event_type": str(event_type),
        "time_seconds": sf(time_value) if time_value is not None else None,
    }


def ordered_response_analysis(anchors: List[Dict[str, Any]]) -> Dict[str, Any]:
    chains = []

    for anchor in anchors:
        events = []
        for item in anchor.get("response_events", []):
            normalized = normalize_response_event(item)
            if normalized and normalized["event_type"] in RESPONSE_EVENT_TYPES:
                events.append(normalized)

        if not events:
            continue

        if all(x["time_seconds"] is not None for x in events):
            events.sort(key=lambda x: x["time_seconds"])

        sequence = []
        for event in events:
            event_type = event["event_type"]
            if not sequence or sequence[-1] != event_type:
                sequence.append(event_type)

        distinct = len(set(sequence))
        chain = {
            "view": anchor.get("view"),
            "interaction_episode_id": anchor.get("interaction_episode_id"),
            "sequence": sequence,
            "distinct_transition_count": distinct,
            "response_event_count": len(events),
            "trigger_score": sf(anchor.get("trigger_score")),
            "response_score": max(
                sf(anchor.get("response_score")),
                sf(anchor.get("strong_response_score")),
            ),
        }
        chains.append(chain)

    best_chain = max(
        chains,
        key=lambda x: (
            x["distinct_transition_count"],
            x["response_event_count"],
            x["response_score"],
        ),
        default=None,
    )

    best_distinct = best_chain["distinct_transition_count"] if best_chain else 0
    total_events = max(
        [chain["response_event_count"] for chain in chains],
        default=0,
    )

    return {
        "chains": chains,
        "best_chain": best_chain,
        "response_event_count": total_events,
        "compound_response": best_distinct >= 2,
        "strong_compound_response": best_distinct >= 3,
    }


def analyze_event(event: Dict[str, Any]) -> Dict[str, Any]:
    anchors = event.get("interaction_anchors", [])
    response = ordered_response_analysis(anchors)

    temporal_consensus = sf(event.get("temporal_consensus"))
    interaction_consensus = sf(event.get("interaction_consensus"))
    max_temporal = sf(event.get("max_temporal_score"))
    max_trigger = sf(event.get("max_trigger_score"))
    max_interaction = sf(event.get("max_interaction_score"))
    max_response = sf(event.get("max_response_score"))

    signal_coherence = sf(event.get("cross_view_signal_coherence"))
    shared_disruptive = event.get("shared_disruptive_signals", [])
    disruptive_views = event.get("disruptive_views", [])

    consequence_views = event.get("consequence_views", [])
    consequence_signal_types = event.get("consequence_signal_types", [])
    consequence_diversity = sf(event.get("consequence_diversity_score"))
    consequence_signals_by_view = event.get("consequence_signals_by_view", {})
    interaction_signal_views = event.get("interaction_signal_views", [])
    scene_activity_views = event.get("scene_activity_views", [])
    scene_only_views = event.get("scene_only_views", [])
    track_disruption_views = event.get("track_disruption_views", [])
    direct_consequence_views = event.get("direct_consequence_views", consequence_views)
    indirect_response_views = event.get("indirect_response_views", [])
    complementary_response_pattern = bool(
        event.get("complementary_response_pattern", False)
    )
    replicated_kinematic_only = bool(
        event.get("replicated_kinematic_only", False)
    )

    scene_activity_dominated = (
        len(scene_activity_views) >= 2
        and len(consequence_views) < MIN_CONSEQUENCE_VIEWS
    )

    heterogeneous_consequence = (
        len(consequence_views) >= MIN_CONSEQUENCE_VIEWS
        and len(consequence_signal_types) >= 2
    )

    consequence_supported = (
        consequence_diversity >= MIN_CONSEQUENCE_DIVERSITY
        and len(consequence_signal_types) >= 3
        and not replicated_kinematic_only
        and (
            len(consequence_views) >= 3
            or (
                len(consequence_views) >= 2
                and len(interaction_signal_views) >= 2
            )
        )
    )

    temporal_views = set(event.get("temporal_views", []))
    interaction_views = set(event.get("interaction_views", []))

    complementary_multiview_response = (
        temporal_consensus >= MIN_TEMPORAL_CONSENSUS
        and max_temporal >= MIN_TEMPORAL_SCORE
        and len(temporal_views) >= 2
        and complementary_response_pattern
        and len(direct_consequence_views) >= 1
        and len(indirect_response_views) >= 1
    )

    critical_anchor_views = {
        anchor.get("view")
        for anchor in anchors
        if anchor.get("current_critical_candidate")
    }

    isolated_projection = (
        len(critical_anchor_views) == 1
        and len(event.get("supporting_views", [])) > 1
        and len(interaction_views) <= 1
    )

    compound = response["compound_response"]
    strong_compound = response["strong_compound_response"]

    route_a = (
        max_trigger >= 0.35
        and max_interaction >= 0.45
        and strong_compound
    )

    route_a_moderate = (
        max_trigger >= 0.35
        and compound
        and max_response >= 0.45
    )

    route_b = (
        temporal_consensus >= MIN_TEMPORAL_CONSENSUS
        and max_temporal >= MIN_TEMPORAL_SCORE
        and len(temporal_views) >= 2
        and consequence_supported
    )

    route_c = complementary_multiview_response

    if isolated_projection and not strong_compound:
        level = 1
        decision = "NON_CRITICAL"
        route = "PROJECTION_AMBIGUITY_SUPPRESSION"
    elif route_a:
        level = 3
        decision = "CRITICAL_CANDIDATE"
        route = "INTERACTION_CONDITIONED_COMPOUND_CONSEQUENCE"
    elif route_a_moderate:
        level = 2
        decision = "AMBIGUOUS"
        route = "INTERACTION_CONDITIONED_RESPONSE"
    elif route_b:
        level = 2
        decision = "VISUAL_CONFIRMATION_REQUIRED"
        route = "CROSS_VIEW_CONSEQUENCE_DIVERSITY"
    elif route_c:
        level = 2
        decision = "VISUAL_CONFIRMATION_REQUIRED"
        route = "COMPLEMENTARY_CROSS_VIEW_RESPONSE"
    elif compound:
        level = 1
        decision = "AMBIGUOUS"
        route = "UNCONDITIONED_COMPOUND_MOTION"
    else:
        level = 0
        decision = "NON_CRITICAL"
        route = "GENERIC_OR_INSUFFICIENT_EVIDENCE"

    reasons = []
    if route == "PROJECTION_AMBIGUITY_SUPPRESSION":
        reasons.append(
            "critical interaction evidence is isolated to one camera projection"
        )
    elif route == "INTERACTION_CONDITIONED_COMPOUND_CONSEQUENCE":
        reasons.extend([
            "ordered multi-transition response chain detected",
            "compound response contains at least three distinct transitions",
            "response is temporally conditioned on an interaction trigger",
        ])
    elif route == "INTERACTION_CONDITIONED_RESPONSE":
        reasons.extend([
            "ordered multi-transition response chain detected",
            "response is temporally conditioned on an interaction trigger",
        ])
    elif route == "CROSS_VIEW_CONSEQUENCE_DIVERSITY":
        reasons.extend([
            "multiple views localize strong abnormality near the same event anchor",
            "entity-consequence signals are present in multiple camera views",
            "different consequence signal types provide heterogeneous multiview support",
        ])
        if track_disruption_views:
            reasons.append("track disruption is present as supporting continuity evidence")
    elif route == "COMPLEMENTARY_CROSS_VIEW_RESPONSE":
        reasons.extend([
            "multiple views localize abnormality near the same event anchor",
            "at least one view contains direct entity-consequence evidence",
            "other synchronized views contain indirect scene-response evidence",
            "complementary camera roles require visual physical-consequence verification",
        ])
    elif route == "UNCONDITIONED_COMPOUND_MOTION":
        reasons.append("ordered multi-transition response chain detected")
    else:
        if replicated_kinematic_only:
            reasons.append(
                "replicated motion/geometry change lacks continuity disruption and remains projection-vulnerable"
            )
        elif scene_activity_dominated:
            reasons.append(
                "synchronized abnormality is dominated by scene-level activity and did not satisfy the complementary-response temporal route"
            )
        elif (
            temporal_consensus >= MIN_TEMPORAL_CONSENSUS
            and max_temporal >= MIN_TEMPORAL_SCORE
            and len(temporal_views) >= 2
            and not consequence_supported
        ):
            reasons.append(
                "cross-view temporal abnormality lacks sufficiently diverse multiview consequence evidence"
            )
        else:
            reasons.append(
                "no compound consequence or multiview consequence-diversity route was satisfied"
            )

    return {
        "fused_event_id": event.get("fused_event_id"),
        "anchor_time": event.get("anchor_time"),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "supporting_views": event.get("supporting_views", []),
        "criticality_level": level,
        "decision": decision,
        "criticality_route": route,
        "final_accident_classification": False,
        "features": {
            "temporal_consensus": round(temporal_consensus, 4),
            "interaction_consensus": round(interaction_consensus, 4),
            "max_temporal_score": round(max_temporal, 4),
            "max_trigger_score": round(max_trigger, 4),
            "max_interaction_score": round(max_interaction, 4),
            "max_response_score": round(max_response, 4),
            "isolated_projection": isolated_projection,
            "response_event_count": response["response_event_count"],
            "compound_response": compound,
            "strong_compound_response": strong_compound,
            "cross_view_signal_coherence": round(signal_coherence, 4),
            "shared_disruptive_signals": shared_disruptive,
            "disruptive_views": disruptive_views,
            "consequence_views": consequence_views,
            "consequence_signal_types": consequence_signal_types,
            "consequence_diversity_score": round(consequence_diversity, 4),
            "consequence_supported": consequence_supported,
            "heterogeneous_consequence": heterogeneous_consequence,
            "interaction_signal_views": interaction_signal_views,
            "scene_activity_views": scene_activity_views,
            "scene_only_views": scene_only_views,
            "scene_activity_dominated": scene_activity_dominated,
            "track_disruption_views": track_disruption_views,
            "direct_consequence_views": direct_consequence_views,
            "indirect_response_views": indirect_response_views,
            "complementary_response_pattern": complementary_response_pattern,
            "replicated_kinematic_only": replicated_kinematic_only,
            "complementary_multiview_response": complementary_multiview_response,
        },
        "best_response_chain": response["best_chain"],
        "response_analysis": response,
        "signal_coherence_analysis": {
            "pairwise_signal_coherence": event.get(
                "pairwise_signal_coherence", []
            ),
            "per_view_signal_signatures": event.get(
                "per_view_signal_signatures", {}
            ),
            "active_signals_by_view": event.get(
                "active_signals_by_view", {}
            ),
            "disruptive_signals_by_view": event.get(
                "disruptive_signals_by_view", {}
            ),
        },
        "reasons": reasons,
    }


def decision_priority(decision: str) -> int:
    return {
        "CRITICAL_CANDIDATE": 4,
        "VISUAL_CONFIRMATION_REQUIRED": 3,
        "AMBIGUOUS": 2,
        "NON_CRITICAL": 1,
    }.get(decision, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fusion_json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.fusion_json.is_file():
        raise FileNotFoundError(args.fusion_json)

    fusion = load_json(args.fusion_json)
    events = [
        analyze_event(event)
        for event in fusion.get("fused_events", [])
    ]

    ranked = sorted(
        events,
        key=lambda x: (
            decision_priority(x["decision"]),
            x["criticality_level"],
            x["features"]["max_temporal_score"],
        ),
        reverse=True,
    )
    for rank, event in enumerate(ranked, start=1):
        event["rank"] = rank

    top = ranked[0] if ranked else None

    output = {
        "configuration": {
            "version": "v5_parallax_aware_semantic_hierarchy",
            "final_accident_classification": False,
            "minimum_consequence_views": MIN_CONSEQUENCE_VIEWS,
            "minimum_consequence_diversity": MIN_CONSEQUENCE_DIVERSITY,
            "minimum_temporal_consensus": MIN_TEMPORAL_CONSENSUS,
            "minimum_temporal_score": MIN_TEMPORAL_SCORE,
            "criticality_levels": {
                "0": "generic or insufficient evidence",
                "1": "isolated/weak motion consequence",
                "2": "interaction response or coherent synchronized disruption requiring verification",
                "3": "interaction-conditioned compound consequence",
                "5": "reserved for future visible physical consequence verification",
            },
            "design_note": (
                "Criticality routing distinguishes direct entity consequence, indirect scene "
                "response, and projection-vulnerable replicated kinematics. Complementary "
                "cross-view roles are routed to visual verification, never directly "
                "declared critical."
            ),
        },
        "case": fusion.get("case"),
        "source_fusion_json": str(args.fusion_json),
        "criticality_events": ranked,
        "summary": {
            "events_analyzed": len(ranked),
            "critical_candidates": sum(
                x["decision"] == "CRITICAL_CANDIDATE" for x in ranked
            ),
            "visual_confirmation_required": sum(
                x["decision"] == "VISUAL_CONFIRMATION_REQUIRED" for x in ranked
            ),
            "ambiguous_events": sum(
                x["decision"] == "AMBIGUOUS" for x in ranked
            ),
            "non_critical_events": sum(
                x["decision"] == "NON_CRITICAL" for x in ranked
            ),
            "top_decision": top["decision"] if top else "NO_EVENT",
            "top_event_id": top["fused_event_id"] if top else None,
            "top_window": (
                {
                    "start_time": top["start_time"],
                    "end_time": top["end_time"],
                }
                if top else None
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 96)
    print("CRITICALITY ANALYZER V5 COMPLETE")
    print("=" * 96)
    print(f"Case                         : {output['case']}")
    print(f"Events analyzed              : {len(ranked)}")
    print(f"Critical candidates          : {output['summary']['critical_candidates']}")
    print(f"Visual confirmation required : {output['summary']['visual_confirmation_required']}")
    print(f"Ambiguous                    : {output['summary']['ambiguous_events']}")
    print(f"Non-critical                 : {output['summary']['non_critical_events']}")
    print(f"Top decision                 : {output['summary']['top_decision']}")
    print(f"Top window                   : {output['summary']['top_window']}")
    print(f"Saved to                     : {args.output}")
    print("=" * 96)

    for event in ranked:
        print(
            f"\nRANK {event['rank']:02d} | {event['fused_event_id']} | "
            f"{event['start_time']:.2f}-{event['end_time']:.2f}s"
        )
        print(f"  decision : {event['decision']}")
        print(f"  level    : {event['criticality_level']}")
        print(f"  route    : {event['criticality_route']}")
        print(f"  compound : {event['features']['compound_response']}")
        print(f"  strong   : {event['features']['strong_compound_response']}")
        print(f"  coherence: {event['features']['cross_view_signal_coherence']:.4f}")
        print(f"  consviews: {event['features']['consequence_views']}")
        print(f"  constypes: {event['features']['consequence_signal_types']}")
        print(f"  consdiv  : {event['features']['consequence_diversity_score']:.4f}")
        print(f"  sceneonly: {event['features']['scene_only_views']}")
        print(f"  direct   : {event['features']['direct_consequence_views']}")
        print(f"  indirect : {event['features']['indirect_response_views']}")
        print(f"  complement: {event['features']['complementary_response_pattern']}")
        print(f"  kin_only : {event['features']['replicated_kinematic_only']}")
        chain = event.get("best_response_chain")
        print(f"  chain    : {chain['sequence'] if chain else []}")
        print(f"  reasons  : {event['reasons']}")


if __name__ == "__main__":
    main()
