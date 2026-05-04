from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .attacks.adv_docvqa_attack import AdvDocVQAAttackConfig, adv_docvqa_attack
from .attacks.diacritics import DiacriticsAttackConfig, diacritics_attack
from .attacks.image_patch import ImagePatchAttackConfig, image_patch_attack
from .attacks.semantic import SemanticAttackConfig, semantic_synonym_attack
from .metrics import cer, normalize_text, wer
from .ocr_engines import ENGINE_RUNNERS
from .render import FreeTypeRenderer, RenderConfig


@dataclass(frozen=True)
class AttackConfig:
    # Text perturbations (operate on string).
    semantic: Optional[SemanticAttackConfig] = None
    diacritics: Optional[DiacriticsAttackConfig] = None
    # Image perturbation (operates on rendered image).
    image_patch: Optional[ImagePatchAttackConfig] = None
    # DocVQA adversarial image perturbation (article 2512.04554v1).
    adv_docvqa: Optional[AdvDocVQAAttackConfig] = None

    # Optional oracle mode for semantic GA (very expensive).
    # If set, semantic GA uses this OCR engine as fitness driver.
    semantic_oracle_engine: Optional[str] = None


@dataclass
class AttackPipeline:
    render_config: RenderConfig
    attack_config: AttackConfig

    def build_attacked_text(self, original_text: str, *, semantic_fitness_fn: Optional[Callable[[str], float]] = None) -> Tuple[str, Dict]:
        text = original_text
        meta: Dict[str, Any] = {}

        if self.attack_config.semantic is not None:
            attacked, m = semantic_synonym_attack(
                text,
                self.attack_config.semantic,
                fitness_fn=semantic_fitness_fn,
            )
            meta["semantic"] = m
            text = attacked

        if self.attack_config.diacritics is not None:
            attacked, m = diacritics_attack(text, self.attack_config.diacritics)
            meta["diacritics"] = m
            text = attacked

        return text, meta

    def render_original(self, text: str, *, record_line_bboxes: bool = False) -> Tuple[Image.Image, Optional[List[Tuple[int, int, int, int]]]]:
        with FreeTypeRenderer(self.render_config) as renderer:
            if record_line_bboxes:
                img, bboxes = renderer.render(text, x=0, y=0, record_line_bboxes=True)
                return img, bboxes
            img = renderer.render(text, x=0, y=0, record_line_bboxes=False)
            return img, None

    def render_attacked(self, text: str) -> Tuple[Image.Image, str, Dict]:
        # Render and apply image-level patches; return attacked_text and meta.
        semantic_oracle_engine = self.attack_config.semantic_oracle_engine
        semantic_fitness_fn: Optional[Callable[[str], float]] = None

        # Prepare oracle fitness if possible (optional, expensive).
        if semantic_oracle_engine is not None and semantic_oracle_engine in ENGINE_RUNNERS:
            runner = ENGINE_RUNNERS[semantic_oracle_engine]

            def fitness_fn(candidate_text: str) -> float:
                # Compute fitness as expected OCR "error" vs ground truth (WER).
                with FreeTypeRenderer(self.render_config) as r:
                    img = r.render(candidate_text, x=0, y=0)
                hyp = runner(img)
                # Higher is better: larger WER => more errors.
                return wer(text, hyp)

            semantic_fitness_fn = fitness_fn

        with FreeTypeRenderer(self.render_config) as renderer:
            attacked_text, meta = self.build_attacked_text(text, semantic_fitness_fn=semantic_fitness_fn)

            img, line_bboxes = renderer.render(attacked_text, x=0, y=0, record_line_bboxes=True)

            if self.attack_config.image_patch is not None and line_bboxes is not None:
                img = image_patch_attack(
                    img,
                    renderer=renderer,
                    line_bboxes=line_bboxes,
                    config=self.attack_config.image_patch,
                )
                meta["image_patch"] = {"effects": list(self.attack_config.image_patch.effects)}

            if self.attack_config.adv_docvqa is not None:
                img, adv_meta = adv_docvqa_attack(img, self.attack_config.adv_docvqa)
                meta["adv_docvqa"] = adv_meta

            return img, attacked_text, meta


def evaluate_ocr_engines(
    *,
    input_text: str,
    pipeline: AttackPipeline,
#    engines: Sequence[str] = ("tesseract", "trocr", "easyocr", "ocr_got"),
    engines: Sequence[str] = tuple(ENGINE_RUNNERS.keys()),
    reference_text: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate OCR error rates (CER/WER) and timings against multiple OCR engines.
    """
    ref = reference_text if reference_text is not None else input_text

    results: Dict[str, Any] = {
        "reference_text": ref,
        "attacked_text": None,
        "metrics": {},
    }

    # Render originals/attacks once; then OCR each engine.
    t0 = time.perf_counter()
    with FreeTypeRenderer(pipeline.render_config) as renderer:
        original_img = renderer.render(input_text, x=0, y=0)
    render_time_original = time.perf_counter() - t0

    t1 = time.perf_counter()
    attacked_img, attacked_text, attack_meta = pipeline.render_attacked(input_text)
    render_time_attacked = time.perf_counter() - t1
    results["attacked_text"] = attacked_text
    results["attack_meta"] = attack_meta

    for engine_name in engines:
        engine_entry: Dict[str, Any] = {"skipped": False}
        runner = ENGINE_RUNNERS.get(engine_name)
        if runner is None:
            engine_entry["skipped"] = True
            engine_entry["error"] = f"Unknown engine: {engine_name}"
            results["metrics"][engine_name] = engine_entry
            continue

        # OCR on original
        try:
            t_ocr0 = time.perf_counter()
            hyp_original = runner(original_img)
            ocr_time_original = time.perf_counter() - t_ocr0
            cer_o = cer(ref, hyp_original)
            wer_o = wer(ref, hyp_original)
        except Exception as e:  # pragma: no cover - depends on external engines
            engine_entry["skipped"] = True
            engine_entry["error"] = f"{type(e).__name__}: {e}"
            results["metrics"][engine_name] = engine_entry
            continue

        # OCR on attacked
        try:
            t_ocr1 = time.perf_counter()
            hyp_attacked = runner(attacked_img)
            ocr_time_attacked = time.perf_counter() - t_ocr1
            cer_a = cer(ref, hyp_attacked)
            wer_a = wer(ref, hyp_attacked)
        except Exception as e:  # pragma: no cover
            engine_entry["skipped"] = True
            engine_entry["error_attacked"] = f"{type(e).__name__}: {e}"
            results["metrics"][engine_name] = engine_entry
            continue

        engine_entry["original"] = {
            "cer": cer_o,
            "wer": wer_o,
            "render_time_sec": render_time_original,
            "ocr_time_sec": ocr_time_original,
            "hypothesis": hyp_original,
        }
        engine_entry["attacked"] = {
            "cer": cer_a,
            "wer": wer_a,
            "render_time_sec": render_time_attacked,
            "ocr_time_sec": ocr_time_attacked,
            "hypothesis": hyp_attacked,
        }

        results["metrics"][engine_name] = engine_entry

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results

