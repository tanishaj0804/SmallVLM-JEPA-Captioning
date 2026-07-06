"""
Fuse a scene caption and V-JEPA 2 action label with TinyLlama.

Standalone:
    python fuse_tinyllama.py \
        --scene "A person is in a kitchen near a counter." \
        --action "Picking something up"
"""

from __future__ import annotations

import argparse
import gc
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class TinyLlamaFuser:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = None
        self.model = None

    def load(self) -> "TinyLlamaFuser":
        if self.model is not None:
            return self

        print(f"[fusion] Loading {self.model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        return self

    @torch.inference_mode()
    def fuse(
        self,
        scene: str,
        action: str,
        action_confidence: Optional[float] = None,
        max_new_tokens: int = 60,
    ) -> str:
        self.load()

        confidence_note = (
            f"\nAction confidence: {action_confidence:.1%}"
            if action_confidence is not None
            else ""
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You fuse two video-analysis signals into one factual caption. "
                    "Return exactly one concise sentence and nothing else. "
                    "Preserve the visible scene from the scene description. "
                    "Use the action label to correct or add temporal motion. "
                    "Do not invent objects, colors, locations, identities, motives, "
                    "or adjectives absent from the inputs. If the action uses the "
                    "word 'something', keep the object generic unless the scene "
                    "clearly identifies the manipulated object."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Scene description: {scene}\n"
                    f"Detected action: {action}"
                    f"{confidence_note}\n"
                    "Write the fused video caption."
                ),
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        generated = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()

        text = " ".join(text.split())
        # Tiny models sometimes prefix the answer despite the instruction.
        for prefix in ("Caption:", "Fused caption:", "Answer:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Keep one sentence if the model rambles.
        for end in (".", "!", "?"):
            pos = text.find(end)
            if pos != -1:
                text = text[: pos + 1]
                break

        return text

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def fuse_caption(
    scene: str,
    action: str,
    action_confidence: Optional[float] = None,
    fuser: Optional[TinyLlamaFuser] = None,
) -> str:
    own_model = fuser is None
    fuser = fuser or TinyLlamaFuser()
    try:
        return fuser.fuse(scene, action, action_confidence)
    finally:
        if own_model:
            fuser.unload()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse scene and action with TinyLlama.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--confidence", type=float, default=None)
    args = parser.parse_args()

    result = fuse_caption(args.scene, args.action, args.confidence)
    print(f"\nFused caption:\n{result}")


if __name__ == "__main__":
    main()
