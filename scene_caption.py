"""
Friend A: scene captioning with SmolVLM2.

Standalone:
    python scene_caption.py path/to/video.mp4
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype(device: torch.device) -> torch.dtype:
    # float32 is the safest CPU fallback. float16 reduces CUDA memory use.
    return torch.float16 if device.type == "cuda" else torch.float32


class SceneCaptioner:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.device = _device()
        self.dtype = _dtype(self.device)
        self.processor = None
        self.model = None

    def load(self) -> "SceneCaptioner":
        if self.model is not None:
            return self

        print(f"[scene] Loading {self.model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        return self

    @torch.inference_mode()
    def caption(
        self,
        video_path: str,
        prompt: str = (
            "Describe the visible scene in one concise sentence. "
            "Focus on the person, objects, and surroundings. "
            "Do not guess fine-grained motion direction if it is unclear."
        ),
        max_new_tokens: int = 80,
    ) -> str:
        self.load()
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(path.resolve())},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # BatchFeature.to(device, dtype) correctly keeps token IDs integral.
        inputs = inputs.to(self.device, dtype=self.dtype)

        generated_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

        # Decode only newly generated tokens where possible.
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        text = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()

        if not text:
            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

        return " ".join(text.split())

    def unload(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def caption_video(video_path: str, captioner: Optional[SceneCaptioner] = None) -> str:
    own_model = captioner is None
    captioner = captioner or SceneCaptioner()
    try:
        return captioner.caption(video_path)
    finally:
        if own_model:
            captioner.unload()


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption a video with SmolVLM2.")
    parser.add_argument("video", help="Path to an MP4/video file")
    args = parser.parse_args()

    caption = caption_video(args.video)
    print(f"\nScene caption:\n{caption}")


if __name__ == "__main__":
    main()
