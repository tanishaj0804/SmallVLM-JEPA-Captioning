
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from action_probe import ActionProbe
from fusion_model import DEFAULT_SCHEMA, EvidenceFuser, load_schema
from scene_caption import SceneCaptioner


def select_actions(
    predictions,
    max_actions: int = 3,
    min_confidence: float = 0.05
):
    selected = [
        prediction
        for prediction in predictions
        if prediction["confidence"] >= min_confidence
    ]

    return selected[:max_actions]


def run_pipeline(
    video_path: str,
    top_k: int = 5,
    num_frames: int = 8,
    keep_loaded: bool = False,
    schema: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict[str, Any]:

    path = Path(video_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Video not found: {path}"
        )

    schema = schema or DEFAULT_SCHEMA

    captioner = SceneCaptioner()
    probe = ActionProbe()
    fuser = EvidenceFuser()

    try:
        print("\n[1/3] Extracting visual evidence...")

        scene_text = captioner.caption(
            str(path),
            num_frames=num_frames
        )

        print("\nScene description:")
        print(scene_text)

        if not keep_loaded:
            captioner.unload()

        print("\n[2/3] Extracting temporal evidence...")

        action_predictions = probe.predict(
            str(path),
            top_k=top_k
        )

        if not action_predictions:
            raise RuntimeError(
                "V-JEPA 2 returned no action predictions."
            )

        print("\nTop action predictions:")

        for item in action_predictions:
            print(
                f"  {item['confidence'] * 100:7.2f}%  "
                f"{item['label']}"
            )

        selected_actions = select_actions(
            action_predictions,
            max_actions=3,
            min_confidence=0.05
        )

        print("\nSelected temporal hypotheses:")

        if selected_actions:
            for item in selected_actions:
                print(
                    f"  {item['confidence'] * 100:7.2f}%  "
                    f"{item['label']}"
                )

        else:
            print(
                "  No temporal hypothesis passed "
                "the confidence threshold."
            )

        if not keep_loaded:
            probe.unload()

        print(
            "\n[3/3] Grounding visual and temporal evidence..."
        )

        structured = fuser.fuse(
            scene=scene_text,
            actions=selected_actions,
            schema=schema
        )

        print("\nFinal structured analysis:")

        print(
            json.dumps(
                structured,
                indent=2,
                ensure_ascii=False
            )
        )

        result = {
            "video": str(path),
            "visual_evidence": scene_text,
            "temporal_predictions": action_predictions,
            "selected_temporal_hypotheses": selected_actions,
            "structured_analysis": structured
        }

        return result

    finally:
        captioner.unload()
        probe.unload()
        fuser.unload()


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run SmolVLM2, V-JEPA 2, and "
            "Qwen2.5 evidence fusion pipeline."
        )
    )

    parser.add_argument(
        "video",
        help="Path to the input video file"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help=(
            "Number of V-JEPA action predictions "
            "to retrieve"
        )
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help=(
            "Number of video frames sampled "
            "for SmolVLM2"
        )
    )

    parser.add_argument(
        "--max_actions",
        type=int,
        default=3,
        help=(
            "Maximum number of temporal hypotheses "
            "used for fusion"
        )
    )

    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.05,
        help=(
            "Minimum V-JEPA confidence required "
            "for fusion"
        )
    )

    parser.add_argument(
        "--keep_loaded",
        action="store_true",
        help=(
            "Keep models loaded until pipeline "
            "completion"
        )
    )

    parser.add_argument(
        "--schema",
        default=None,
        help="Optional JSON schema file"
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON file path"
    )

    args = parser.parse_args()

    schema = (
        load_schema(args.schema)
        if args.schema
        else None
    )

    path = Path(args.video)

    if not path.is_file():
        raise FileNotFoundError(
            f"Video not found: {path}"
        )

    schema = schema or DEFAULT_SCHEMA

    captioner = SceneCaptioner()
    probe = ActionProbe()
    fuser = EvidenceFuser()

    try:
        print("\n[1/3] Extracting visual evidence...")

        scene_text = captioner.caption(
            str(path),
            num_frames=args.num_frames
        )

        print("\nScene description:")
        print(scene_text)

        if not args.keep_loaded:
            captioner.unload()

        print("\n[2/3] Extracting temporal evidence...")

        action_predictions = probe.predict(
            str(path),
            top_k=args.top_k
        )

        if not action_predictions:
            raise RuntimeError(
                "V-JEPA 2 returned no action predictions."
            )

        print("\nTop action predictions:")

        for item in action_predictions:
            print(
                f"  {item['confidence'] * 100:7.2f}%  "
                f"{item['label']}"
            )

        selected_actions = select_actions(
            action_predictions,
            max_actions=args.max_actions,
            min_confidence=args.min_confidence
        )

        print("\nSelected temporal hypotheses:")

        if selected_actions:
            for item in selected_actions:
                print(
                    f"  {item['confidence'] * 100:7.2f}%  "
                    f"{item['label']}"
                )

        else:
            print(
                "  No temporal hypothesis passed "
                "the confidence threshold."
            )

        if not args.keep_loaded:
            probe.unload()

        print(
            "\n[3/3] Grounding visual and temporal evidence..."
        )

        structured = fuser.fuse(
            scene=scene_text,
            actions=selected_actions,
            schema=schema
        )

        print("\nFinal structured analysis:")

        print(
            json.dumps(
                structured,
                indent=2,
                ensure_ascii=False
            )
        )

        result = {
            "video": str(path),
            "visual_evidence": scene_text,
            "temporal_predictions": action_predictions,
            "selected_temporal_hypotheses": selected_actions,
            "structured_analysis": structured
        }

    finally:
        captioner.unload()
        probe.unload()
        fuser.unload()

    out_path = (
        Path(args.out)
        if args.out
        else Path("results") / f"{path.stem}.json"
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print(f"Result written to: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
