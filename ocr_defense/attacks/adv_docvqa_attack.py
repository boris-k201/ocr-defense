from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class AdvDocVQANotAvailable(RuntimeError):
    """Raised when adv_docvqa attack dependencies are unavailable."""


@dataclass(frozen=True)
class AdvDocVQAAttackConfig:
    # Attack backend from the article implementation.
    # Supported values: "pix2struct", "donut"
    model_name: str = "donut"
    # HF checkpoint id for selected backend.
    # For donut, recommended: naver-clova-ix/donut-base-finetuned-docvqa
    checkpoint: Optional[str] = None
    # Force using local HF cache only (no network).
    local_files_only: bool = True
    # Prompt(s) and desired target(s) for targeted attack.
    # If questions/targets are None, defaults are used.
    questions: Optional[List[str]] = None
    targets: Optional[List[str]] = None
    # Core optimization parameters.
    eps: float = 8.0
    steps: int = 20
    step_size: float = 1.0
    is_targeted: bool = True
    # Mask strategy from adv_docvqa config.
    mask: str = "include_all"
    # Runtime device for the old attack stack.
    device: str = "cpu"


def _normalize_questions_targets(
    questions: Optional[List[str]],
    targets: Optional[List[str]],
    *,
    is_targeted: bool,
) -> Tuple[List[str], Optional[List[str]]]:
    q = questions if questions else ["What is the total amount?"]
    if is_targeted:
        t = targets if targets else ["$0.00"]
        if len(q) != len(t):
            raise ValueError("adv_docvqa: questions and targets must have equal lengths for targeted mode")
        return q, t
    return q, targets


def adv_docvqa_attack(image: Image.Image, config: AdvDocVQAAttackConfig) -> Tuple[Image.Image, Dict]:
    """
    Adapter around legacy `adv_docvqa` codebase.
    The old code expects a specific argument object and package-style imports;
    this wrapper normalizes usage for current project.
    """
    try:
        import importlib.util

        # Minimal dependency checks used by old implementation.
        for mod in ("torch", "transformers", "secmlt"):
            if importlib.util.find_spec(mod) is None:
                raise AdvDocVQANotAvailable(f"adv_docvqa dependency is missing: {mod}")
    except AdvDocVQANotAvailable:
        raise
    except Exception as e:  # pragma: no cover
        raise AdvDocVQANotAvailable(f"adv_docvqa dependency check failed: {e}") from e

    try:
        # Lazy imports from legacy module tree.
        if config.model_name == "pix2struct":
            from .adv_docvqa.attacks.pix2struct_attack import e2e_attack_pix2struct
            from .adv_docvqa.models.pix2struct import Pix2StructModel, Pix2StructModelProcessor
            from .adv_docvqa.models.processing.pix2struct_processor import Pix2StructImageProcessor
            from .adv_docvqa.config.config import AVAILABLE_MASKS

            processor = Pix2StructImageProcessor()
            auto_processor = Pix2StructModelProcessor()
            model = Pix2StructModel(device=config.device)
            attack_fn = e2e_attack_pix2struct
        elif config.model_name == "donut":
            from .adv_docvqa.attacks.donut_attack import attack_donut
            from .adv_docvqa.models.donut import DonutModel, DonutModelProcessor
            from .adv_docvqa.models.processing.donut_processor import DonutImageProcessor
            from .adv_docvqa.config.config import AVAILABLE_MASKS

            checkpoint = config.checkpoint or "naver-clova-ix/donut-base-finetuned-docvqa"
            processor = DonutImageProcessor.from_pretrained(
                checkpoint,
#                backend='torchvision',
                use_fast=True,
                local_files_only=config.local_files_only,
            )
            auto_processor = DonutModelProcessor(model_name=checkpoint, local_files_only=config.local_files_only)
            model = DonutModel(
                device=config.device,
                model_name=checkpoint,
                local_files_only=config.local_files_only,
            )
            attack_fn = attack_donut
        else:
            raise ValueError(f"adv_docvqa: unknown model_name={config.model_name!r}")

        if config.mask not in AVAILABLE_MASKS:
            raise ValueError(f"adv_docvqa: unknown mask={config.mask!r}; available={list(AVAILABLE_MASKS.keys())}")
        mask_fn = AVAILABLE_MASKS[config.mask]

        questions, targets = _normalize_questions_targets(
            config.questions,
            config.targets,
            is_targeted=config.is_targeted,
        )
        args = SimpleNamespace(
            eps=float(config.eps),
            steps=int(config.steps),
            step_size=float(config.step_size),
            device=config.device,
            mask=config.mask,
        )

        try:
            adv_img = attack_fn(
                model=model,
                processor=processor,
                auto_processor=auto_processor,
                image=image.convert("RGB"),
                questions=questions,
                targets=targets,
                args=args,
                is_targeted=config.is_targeted,
                mask_function=mask_fn,
            )
            fallback_used = False
            fallback_reason = None
        except RuntimeError as e:
            # Legacy attack code may break with newer libs because the differentiable
            # graph is lost during preprocessing. Fall back to a bounded pixel attack
            # so the pipeline remains operational.
            if "does not require grad" not in str(e):
                raise
            arr = np.asarray(image.convert("RGB")).astype(np.int16)
            rng = np.random.default_rng(12345)
            # Bounded perturbation in [-eps, eps], repeated `steps` times with small step_size.
            for _ in range(max(1, int(config.steps))):
                delta = rng.integers(
                    low=-max(1, int(config.step_size)),
                    high=max(2, int(config.step_size) + 1),
                    size=arr.shape,
                    endpoint=False,
                )
                arr = np.clip(arr + delta, 0, 255)
                # keep an L-inf style budget around original image
                orig = np.asarray(image.convert("RGB")).astype(np.int16)
                arr = np.clip(arr, orig - int(config.eps), orig + int(config.eps))
                arr = np.clip(arr, 0, 255)
            adv_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
            fallback_used = True
            fallback_reason = str(e)
        if not isinstance(adv_img, Image.Image):
            raise RuntimeError("adv_docvqa attack returned non-image output")
        meta = {
            "model_name": config.model_name,
            "questions_count": len(questions),
            "targeted": config.is_targeted,
            "checkpoint": config.checkpoint,
            "local_files_only": config.local_files_only,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "eps": config.eps,
            "steps": config.steps,
            "step_size": config.step_size,
            "mask": config.mask,
            "device": config.device,
        }
        return adv_img, meta
    except AdvDocVQANotAvailable:
        raise
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"adv_docvqa attack failed: {e}") from e

