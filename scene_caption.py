"""
Detailed visual scene description using SmolVLM2.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"


SCENE_PROMPT = SCENE_PROMPT = """
Carefully observe all of the supplied video frames.

Describe the visible scene in one detailed paragraph.

Include the type of place or environment, visible people, visible vehicles,
important objects, clearly visible colors, the foreground and background,
and the spatial arrangement of major objects.

Compare the earlier and later frames and describe visible activity or
changes across the frames.

Also describe any clearly visible conditions that may be relevant to safety.
Examples include pedestrians close to a roadway, dense vehicle traffic,
people moving among vehicles, vehicles close to pedestrians, blocked paths,
or other directly visible conditions.

Describe safety-related observations only when they are clearly visible.

Only describe what can be supported by the supplied frames.

Do not invent gender, clothing details, bags, accessories, identities,
intentions, or exact actions that are not clearly visible.

Do not predict accidents.

Do not say that someone is in danger.

Do not assign a risk level.

Do not infer causes or intentions.

Write a natural detailed paragraph of approximately 6 to 10 sentences.

Do not return JSON.
Do not use bullet points.
""".strip()


def _device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def _dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16

    return torch.float32


def _load_video_frames(
    path: Path,
    num_frames: int = 8,
) -> np.ndarray:

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video: {path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        frames = []

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frames.append(frame)

        cap.release()

        if not frames:
            raise RuntimeError(
                f"Could not decode video: {path}"
            )

        indices = np.linspace(
            0,
            len(frames) - 1,
            num=min(num_frames, len(frames)),
            dtype=int,
        )

        selected_frames = [
            frames[index]
            for index in indices
        ]

        return np.stack(selected_frames)

    indices = np.linspace(
        0,
        total_frames - 1,
        num=min(num_frames, total_frames),
        dtype=int,
    )

    frames = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index),
        )

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(
            f"Could not decode video: {path}"
        )

    return np.stack(frames)


class SceneCaptioner:

    def __init__(
        self,
        model_id: str = MODEL_ID,
    ) -> None:

        self.model_id = model_id

        self.device = _device()

        self.dtype = _dtype(
            self.device
        )

        self.processor = None

        self.model = None


    def load(self) -> "SceneCaptioner":

        if self.model is not None:
            return self

        print(
            f"[scene] Loading "
            f"{self.model_id} "
            f"on {self.device}..."
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                self.model_id
            )
        )

        self.model = (
            AutoModelForImageTextToText
            .from_pretrained(
                self.model_id,
                dtype=self.dtype,
            )
            .to(self.device)
        )

        self.model.eval()

        return self


    @torch.inference_mode()
    def caption(
        self,
        video_path: str,
        num_frames: int = 8,
        max_new_tokens: int = 300,
    ) -> str:

        self.load()

        path = Path(video_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Video not found: {path}"
            )

        frames = _load_video_frames(
            path,
            num_frames=num_frames,
        )

        frame_images = [
            Image.fromarray(frame)
            for frame in frames
        ]

        content = []

        for image in frame_images:

            content.append(
                {
                    "type": "image",
                    "image": image,
                }
            )

        content.append(
            {
                "type": "text",
                "text": SCENE_PROMPT,
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = (
            self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        inputs = inputs.to(
            self.device,
            dtype=self.dtype,
        )

        generated_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

        input_length = (
            inputs["input_ids"].shape[1]
        )

        new_tokens = generated_ids[
            :,
            input_length:
        ]

        text = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )[0].strip()

        if not text:

            text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()

        return " ".join(
            text.split()
        )


    def unload(self) -> None:

        self.model = None

        self.processor = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def caption_video(
    video_path: str,
    captioner: Optional[
        SceneCaptioner
    ] = None,
    num_frames: int = 8,
) -> str:

    own_model = captioner is None

    captioner = (
        captioner
        or SceneCaptioner()
    )

    try:

        return captioner.caption(
            video_path,
            num_frames=num_frames,
        )

    finally:

        if own_model:
            captioner.unload()


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Describe video frames "
            "using SmolVLM2."
        )
    )

    parser.add_argument(
        "video",
        help="Path to video file",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help=(
            "Number of uniformly "
            "sampled video frames"
        ),
    )

    args = parser.parse_args()

    description = caption_video(
        args.video,
        num_frames=args.num_frames,
    )

    print(
        "\nScene description:\n"
    )

    print(description)


if __name__ == "__main__":
    main()
