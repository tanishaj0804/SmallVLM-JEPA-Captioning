"""
Three-model video caption fusion pipeline.

Default sequential mode:
    SmolVLM2 -> unload -> V-JEPA 2 -> unload -> TinyLlama -> unload

Usage:
    python pipeline.py path/to/video.mp4
    python pipeline.py path/to/video.mp4 --top_k 5
    python pipeline.py path/to/video.mp4 --keep_loaded
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from action_probe import ActionProbe
from fuse_tinyllama import TinyLlamaFuser
from scene_caption import SceneCaptioner


def run_pipeline(
    video_path: str,
    top_k: int = 5,
    keep_loaded: bool = False,
) -> Dict[str, Any]:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    captioner = SceneCaptioner()
    probe = ActionProbe()
    fuser = TinyLlamaFuser()

    try:
        print("\n[1/3] Understanding the visible scene...")
        scene = captioner.caption(str(path))
        print(f"Scene: {scene}")
        if not keep_loaded:
            captioner.unload()

        print("\n[2/3] Understanding temporal action...")
        action_predictions = probe.predict(str(path), top_k=top_k)
        if not action_predictions:
            raise RuntimeError("V-JEPA 2 returned no action predictions.")

        best_action = action_predictions[0]
        print("Top actions:")
        for item in action_predictions:
            print(f"  {item['confidence'] * 100:7.2f}%  {item['label']}")
        if not keep_loaded:
            probe.unload()

        print("\n[3/3] Fusing scene and motion...")
        fused = fuser.fuse(
            scene=scene,
            action=best_action["label"],
            action_confidence=best_action["confidence"],
        )
        print(f"Fused: {fused}")

        return {
            "video": str(path),
            "scene_caption": scene,
            "action": best_action,
            "top_actions": action_predictions,
            "fused_caption": fused,
        }
    finally:
        captioner.unload()
        probe.unload()
        fuser.unload()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse SmolVLM2 scene understanding with V-JEPA 2 motion."
    )
    parser.add_argument("video", help="Path to an MP4/video file")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--keep_loaded",
        action="store_true",
        help="Keep models resident until the pipeline finishes.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final result as JSON.",
    )
    args = parser.parse_args()

    result = run_pipeline(
        video_path=args.video,
        top_k=args.top_k,
        keep_loaded=args.keep_loaded,
    )

    print("\n" + "=" * 72)
    print("FINAL CAPTION")
    print("=" * 72)
    print(result["fused_caption"])

    if args.json:
        print("\nJSON RESULT")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
