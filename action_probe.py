"""
Friend B: motion/action classification with the V-JEPA 2 SSv2 probe.

Standalone:
    python action_probe.py path/to/video.mp4
    python action_probe.py path/to/video.mp4 --top_k 5
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torchcodec.decoders import VideoDecoder
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"


class ActionProbe:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

    def load(self) -> "ActionProbe":
        if self.model is not None:
            return self

        print(f"[action] Loading {self.model_id} on {self.device}...")
        self.processor = AutoVideoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVideoClassification.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()
        return self

    @staticmethod
    def _sample_indices(total_frames: int, num_frames: int) -> np.ndarray:
        if total_frames <= 0:
            raise ValueError("The video contains no decodable frames.")
        # Uniform temporal sampling preserves clip-wide motion better than
        # taking only the first frames.
        return np.linspace(
            0, total_frames - 1, num=num_frames, dtype=np.int64
        )

    @torch.inference_mode()
    def predict(self, video_path: str, top_k: int = 5) -> List[Dict[str, float]]:
        self.load()
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")

        decoder = VideoDecoder(str(path.resolve()))
        total_frames = len(decoder)
        num_frames = int(getattr(self.model.config, "frames_per_clip", 16))
        indices = self._sample_indices(total_frames, num_frames)

        # torchcodec returns T x C x H x W, which AutoVideoProcessor accepts.
        video = decoder.get_frames_at(indices=indices.tolist()).data
        inputs = self.processor(video, return_tensors="pt").to(self.device)

        logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

        k = min(top_k, probabilities.numel())
        probs, class_ids = probabilities.topk(k)

        results: List[Dict[str, float]] = []
        for class_id, probability in zip(class_ids.tolist(), probs.tolist()):
            label = self.model.config.id2label[class_id]
            results.append(
                {
                    "label": str(label),
                    "confidence": float(probability),
                }
            )
        return results

    def unload(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def classify_action(
    video_path: str,
    top_k: int = 5,
    probe: Optional[ActionProbe] = None,
) -> List[Dict[str, float]]:
    own_model = probe is None
    probe = probe or ActionProbe()
    try:
        return probe.predict(video_path, top_k=top_k)
    finally:
        if own_model:
            probe.unload()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify video motion with the V-JEPA 2 SSv2 probe."
    )
    parser.add_argument("video", help="Path to an MP4/video file")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    predictions = classify_action(args.video, top_k=args.top_k)
    print("\nTop action predictions:")
    for item in predictions:
        print(f"{item['confidence'] * 100:7.2f}%   {item['label']}")


if __name__ == "__main__":
    main()
