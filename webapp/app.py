from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ocr_defense.attacks.diacritics import DiacriticsAttackConfig
from ocr_defense.attacks.adv_docvqa_attack import AdvDocVQAAttackConfig
from ocr_defense.attacks.image_patch import ImagePatchAttackConfig
from ocr_defense.attacks.semantic import SemanticAttackConfig
from ocr_defense.evaluation import AttackConfig, AttackPipeline, evaluate_ocr_engines
from ocr_defense.render import RenderConfig


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="OCR Defense Demo")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


AttackName = Literal["none", "semantic", "diacritics", "image_patch", "adv_docvqa", "all"]


class RenderOptions(BaseModel):
    image_width: int = 800
    image_height: int = 600
    margin: int = 10
    font_path: Optional[str] = None
    font_size: int = 22
    dpi: int = 96
    text_color: str = "#000000"
    background_color: str = "#FFFFFF"


class SemanticOptions(BaseModel):
    enabled: bool = False
    language: str = "auto"
    max_changed_words: int = 3
    population_size: int = 24
    generations: int = 18
    random_seed: Optional[int] = None


class DiacriticsOptions(BaseModel):
    enabled: bool = False
    budget_per_word: int = 5
    diacritics_probability: float = 0.6
    random_seed: Optional[int] = None


class ImagePatchOptions(BaseModel):
    enabled: bool = False
    max_patches_per_line: int = 1
    random_seed: Optional[int] = None


class AdvDocVQAOptions(BaseModel):
    enabled: bool = False
    model_name: str = "donut"
    checkpoint: str = "naver-clova-ix/donut-base-finetuned-docvqa"
    local_files_only: bool = True
    # Comma-separated values in UI are normalized server-side.
    questions: Optional[List[str]] = None
    targets: Optional[List[str]] = None
    eps: float = 8.0
    steps: int = 20
    step_size: float = 1.0
    is_targeted: bool = True
    mask: str = "include_all"
    device: str = "cpu"


class AdvancedAttackOptions(BaseModel):
    semantic: SemanticOptions = Field(default_factory=SemanticOptions)
    diacritics: DiacriticsOptions = Field(default_factory=DiacriticsOptions)
    image_patch: ImagePatchOptions = Field(default_factory=ImagePatchOptions)
    adv_docvqa: AdvDocVQAOptions = Field(default_factory=AdvDocVQAOptions)
    semantic_oracle_engine: Optional[str] = None


class RenderRequest(BaseModel):
    text: str
    render: RenderOptions = Field(default_factory=RenderOptions)
    attack: AttackName = "none"
    advanced: AdvancedAttackOptions = Field(default_factory=AdvancedAttackOptions)


class EvaluateRequest(BaseModel):
    text: str
    engines: List[str] = Field(default_factory=lambda: ["tesseract", "trocr", "donut", "easyocr"])
    render: RenderOptions = Field(default_factory=RenderOptions)
    attack: AttackName = "all"
    advanced: AdvancedAttackOptions = Field(default_factory=AdvancedAttackOptions)


def _build_attack_config(attack: AttackName, adv: AdvancedAttackOptions) -> AttackConfig:
    if attack == "none":
        return AttackConfig()

    semantic_cfg = None
    diacritics_cfg = None
    image_patch_cfg = None
    adv_docvqa_cfg = None

    if attack in ("semantic", "all") and adv.semantic.enabled:
        semantic_cfg = SemanticAttackConfig(
            language=adv.semantic.language,
            max_changed_words=adv.semantic.max_changed_words,
            population_size=adv.semantic.population_size,
            generations=adv.semantic.generations,
            random_seed=adv.semantic.random_seed,
        )
    if attack in ("diacritics", "all") and adv.diacritics.enabled:
        diacritics_cfg = DiacriticsAttackConfig(
            budget_per_word=adv.diacritics.budget_per_word,
            diacritics_probability=adv.diacritics.diacritics_probability,
            random_seed=adv.diacritics.random_seed,
        )
    if attack in ("image_patch", "all") and adv.image_patch.enabled:
        image_patch_cfg = ImagePatchAttackConfig(
            max_patches_per_line=adv.image_patch.max_patches_per_line,
            random_seed=adv.image_patch.random_seed,
        )
    if attack in ("adv_docvqa", "all") and adv.adv_docvqa.enabled:
        adv_docvqa_cfg = AdvDocVQAAttackConfig(
            model_name=adv.adv_docvqa.model_name,
            checkpoint=adv.adv_docvqa.checkpoint,
            local_files_only=adv.adv_docvqa.local_files_only,
            questions=adv.adv_docvqa.questions,
            targets=adv.adv_docvqa.targets,
            eps=adv.adv_docvqa.eps,
            steps=adv.adv_docvqa.steps,
            step_size=adv.adv_docvqa.step_size,
            is_targeted=adv.adv_docvqa.is_targeted,
            mask=adv.adv_docvqa.mask,
            device=adv.adv_docvqa.device,
        )

    return AttackConfig(
        semantic=semantic_cfg,
        diacritics=diacritics_cfg,
        image_patch=image_patch_cfg,
        adv_docvqa=adv_docvqa_cfg,
        semantic_oracle_engine=adv.semantic_oracle_engine,
    )


def _render_config_from_opts(opts: RenderOptions) -> RenderConfig:
    return RenderConfig(
        image_width=opts.image_width,
        image_height=opts.image_height,
        margin=opts.margin,
        font_path=opts.font_path,
        font_size=opts.font_size,
        dpi=opts.dpi,
        text_color=opts.text_color,
        background_color=opts.background_color,
    )


def _img_to_data_url_png(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


@app.get("/", response_class=HTMLResponse)
def page_render(request: Request):
    return templates.TemplateResponse(request=request, name="render.html", context={})


@app.get("/testing", response_class=HTMLResponse)
def page_testing(request: Request):
    return templates.TemplateResponse(request=request, name="testing.html", context={})


@app.post("/api/render")
def api_render(payload: RenderRequest):
    render_cfg = _render_config_from_opts(payload.render)
    attack_cfg = _build_attack_config(payload.attack, payload.advanced)
    pipeline = AttackPipeline(render_config=render_cfg, attack_config=attack_cfg)

    if payload.attack == "none":
        img, _ = pipeline.render_original(payload.text, record_line_bboxes=False)
        attacked_text = payload.text
        attack_meta: Dict[str, Any] = {}
    else:
        img, attacked_text, attack_meta = pipeline.render_attacked(payload.text)

    return JSONResponse(
        {
            "image_data_url": _img_to_data_url_png(img),
            "attacked_text": attacked_text,
            "attack_meta": attack_meta,
        }
    )


@app.post("/api/evaluate")
def api_evaluate(payload: EvaluateRequest):
    render_cfg = _render_config_from_opts(payload.render)
    attack_cfg = _build_attack_config(payload.attack, payload.advanced)
    pipeline = AttackPipeline(render_config=render_cfg, attack_config=attack_cfg)

    results = evaluate_ocr_engines(
        input_text=payload.text,
        pipeline=pipeline,
        engines=payload.engines,
        reference_text=payload.text,
        output_path=None,
    )
    return JSONResponse(results)

