"""
Evidence fusion using Qwen2.5-3B-Instruct with -bit QUantization
"""

from __future__ import annotations

import gc
import json
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

DEFAULT_SCHEMA = {
    "scene_summary": {
        "type": "string",
        "description": "Concise factual summary of the scene."
    },
    "people_present": {
        "type": "boolean",
        "description": "Whether people are explicitly visible."
    },
    "people_description": {
        "type": "string",
        "description": "Supported description of visible people."
    },
    "objects": {
        "type": "list",
        "description": "Visible objects, vehicles, and scene elements."
    },
    "environment": {
        "type": "string",
        "description": "Visible environment and setting."
    },
    "grounded_actions": {
        "type": "list",
        "description": "Actions supported by visual and temporal evidence."
    },
    "unresolved_motion": {
        "type": "list",
        "description": "Motion predictions that cannot be assigned confidently."
    },
    "safety_concerns": {
        "type": "list",
        "description": "Potential safety concerns supported by evidence."
    },
    "risk_level": {
        "type": "string",
        "description": "low, moderate, high, or unknown."
    },
    "notable_details": {
        "type": "string",
        "description": "Other important supported details."
    }
}

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_schema(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _format_actions(actions: List[Dict[str, Any]]) -> str:
    if not actions:
        return "No temporal hypotheses available."

    lines = []

    for rank, action in enumerate(actions, start=1):
        label = str(action["label"]).strip()
        confidence = float(action["confidence"])

        lines.append(
            f"{rank}. {label} | confidence={confidence:.2%}"
        )

    return "\n".join(lines)


def _build_prompt(
    scene: str,
    actions: List[Dict[str, Any]]
) -> str:
    return f"""
Fuse visual and temporal evidence from a video.

VISUAL EVIDENCE:
{scene}

TEMPORAL HYPOTHESES:
{_format_actions(actions)}

Return exactly one JSON object with this structure:

{{
  "scene_summary": "",
  "people_present": false,
  "people_description": "",
  "objects": [],
  "environment": "",
  "grounded_actions": [],
  "unresolved_motion": [],
  "safety_concerns": [],
  "risk_level": "",
  "notable_details": ""
}}

Rules:

Use the visual evidence to identify visible people, vehicles,
objects, environment, and directly described activities.

Temporal hypotheses describe possible motion patterns.

Labels containing [something] or [part] do not identify the
moving subject or object.

Temporal hypotheses are ranked alternatives. Do not assume all
predictions happen simultaneously.

Ground a temporal action only when the visual evidence contains
a clearly compatible entity.

If multiple visible entities could match a temporal placeholder,
do not choose one. Preserve the motion in unresolved_motion.

Higher-confidence temporal hypotheses are stronger evidence.

Do not transfer action confidence scores to objects or people.

Do not invent gender, clothing, bags, identities, intentions,
vehicle details, object details, or exact counts.

Ignore vague scene adjectives such as vibrant, lively,
dynamic, and bustling when performing factual reasoning.

Do not use high speed or speeding as safety evidence unless
supported independently by temporal evidence.

Safety concerns must follow from supported evidence.

Pedestrian and vehicle activity in the same road environment
may be described as a potential interaction concern.

Do not predict accidents or claim that somebody will be harmed.

risk_level must be exactly:
low, moderate, high, or unknown.

Use moderate when evidence shows meaningful interaction between
moving vehicles, pedestrians, or potentially conflicting motion.

Use high only for a clearly described immediate dangerous event.

Every list must contain strings only.

Return JSON only.
Do not use markdown.
Do not explain the output.
""".strip()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    match = _JSON_RE.search(text)

    if not match:
        return None

    candidate = match.group(0)

    try:
        return json.loads(candidate)

    except json.JSONDecodeError:
        candidate = re.sub(
            r",\s*([}\]])",
            r"\1",
            candidate
        )

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            return None


def _normalize(parsed: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "scene_summary": "",
        "people_present": False,
        "people_description": "",
        "objects": [],
        "environment": "",
        "grounded_actions": [],
        "unresolved_motion": [],
        "safety_concerns": [],
        "risk_level": "unknown",
        "notable_details": ""
    }

    string_fields = [
        "scene_summary",
        "people_description",
        "environment",
        "notable_details"
    ]

    list_fields = [
        "objects",
        "grounded_actions",
        "unresolved_motion",
        "safety_concerns"
    ]

    for field in string_fields:
        value = parsed.get(field, "")

        if isinstance(value, str):
            result[field] = value.strip()

    people_present = parsed.get(
        "people_present",
        False
    )

    if isinstance(people_present, bool):
        result["people_present"] = people_present

    for field in list_fields:
        value = parsed.get(field, [])

        if isinstance(value, list):
            result[field] = [
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            ]

    risk_level = str(
        parsed.get("risk_level", "unknown")
    ).strip().lower()

    if risk_level in {
        "low",
        "moderate",
        "high",
        "unknown"
    }:
        result["risk_level"] = risk_level

    return result


class EvidenceFuser:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None


    def load(self) -> "EvidenceFuser":
        if self.model is not None:
            return self

        print(
            f"[fusion] Loading {self.model_id} "
            "in 4-bit on CUDA..."
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for the current 4-bit fusion setup."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map={"": 0},
            low_cpu_mem_usage=True
        )

        self.model.eval()

        return self


    @torch.inference_mode()
    def _generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evidence-grounded video analysis "
                    "fusion system. Preserve uncertainty and return "
                    "valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_ids = generated_ids[
            :,
            inputs["input_ids"].shape[1]:
        ]

        return self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0].strip()


    def fuse(
        self,
        scene: str,
        actions: List[Dict[str, Any]],
        schema=None
    ) -> Dict[str, Any]:
        self.load()

        prompt = _build_prompt(
            scene,
            actions
        )

        raw_text = self._generate(prompt)

        parsed = _extract_json(raw_text)

        if parsed is None:
            raise RuntimeError(
                "Qwen fusion model returned invalid JSON.\n\n"
                f"Raw output:\n{raw_text}"
            )

        return _normalize(parsed)


    def unload(self) -> None:
        self.model = None
        self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def fuse_caption(
    scene: str,
    actions: List[Dict[str, Any]],
    schema=None,
    fuser: Optional[EvidenceFuser] = None
) -> Dict[str, Any]:
    own_model = fuser is None

    fuser = fuser or EvidenceFuser()

    try:
        return fuser.fuse(
            scene=scene,
            actions=actions,
            schema=schema
        )

    finally:
        if own_model:
            fuser.unload()
