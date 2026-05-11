import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# До импорта Paddle: отключить проверку доступности хостов моделей (шум и задержка).
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# Меньше предупреждений от tokenizers при многопоточности.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Тише прогресс-бары Hugging Face при загрузке весов.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from ocr_defense.attacks.diacritics import DiacriticsAttackConfig
from ocr_defense.attacks.adv_docvqa_attack import AdvDocVQAAttackConfig
from ocr_defense.attacks.distortions import DistortionsAttackConfig
from ocr_defense.attacks.image_patch import ImagePatchAttackConfig
from ocr_defense.attacks.semantic import SemanticAttackConfig
from ocr_defense.attacks.watermark import WatermarkAttackConfig
from ocr_defense.evaluation import AttackConfig, AttackPipeline, evaluate_ocr_engines
from ocr_defense.ocr_engines import ENGINE_RUNNERS
from ocr_defense.render import DEFAULT_RENDER_CONFIG, FreeTypeRenderer, RenderConfig

def read_text(text_path: str) -> str:
    if text_path == "-":
        return sys.stdin.read()
    input_path = Path(text_path)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    return input_path.read_text(encoding="utf-8")

def read_texts(text_path: str, separator: str, *, skip_empty: bool) -> List[str]:
    raw = read_text(text_path)
    parts = raw.split(separator)
    if skip_empty:
        return [p.strip() for p in parts if p.strip()]
    return [p.strip() for p in parts]

def _tuple2(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    return default

def _tuple_str(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        items = tuple(str(v) for v in value if str(v).strip())
        return items if items else default
    return default

def load_app_config(config_path: Path) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "render": {
            "image_width": DEFAULT_RENDER_CONFIG.image_width,
            "image_height": DEFAULT_RENDER_CONFIG.image_height,
            "margin": DEFAULT_RENDER_CONFIG.margin,
            "font_path": DEFAULT_RENDER_CONFIG.font_path,
            "font_size": DEFAULT_RENDER_CONFIG.font_size,
            "dpi": DEFAULT_RENDER_CONFIG.dpi,
            "text_color": DEFAULT_RENDER_CONFIG.text_color,
            "background_color": DEFAULT_RENDER_CONFIG.background_color,
        },
        "attack": {
            "semantic": {
                "language": "auto",
                "max_changed_words": 3,
                "population_size": 24,
                "generations": 18,
                "tournament_k": 3,
                "mutation_rate": 0.15,
                "crossover_rate": 0.7,
                "random_seed": None,
            },
            "diacritics": {
                "budget_per_word": 5,
                "diacritics_probability": 0.6,
                "random_seed": None,
            },
            "image_patch": {
                "max_patches_per_line": 1,
                "patch_width_ratio": [0.2, 0.6],
                "patch_height_ratio": [0.2, 0.7],
                "pixel_value_min": 40,
                "pixel_value_max": 220,
                "pixel_fill_mode": "random",
                "effects": ["bbox", "pixel", "text"],
                "text_charset": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "patch_text_length_range": [3, 8],
                "patch_font_size": None,
                "random_seed": None,
            },
            "watermark": {
                "text_lines": ["CONFIDENTIAL"],
                "color": "#606060",
                "alpha": 80,
                "font_path": None,
                "font_size": 24,
                "x_spacing": 160,
                "y_spacing": 120,
                "angle_deg": -18.0,
                "x_offset": 0,
                "y_offset": 0,
            },
            "distortions": {
                "enable_skew": True,
                "enable_rotate": True,
                "enable_warp": True,
                "enable_strikethrough": True,
                "character_distort_probability": 0.18,
                "skew_degrees": 10.0,
                "rotate_degrees": 9.0,
                "warp_probability": 1.0,
                "warp_amplitude": 2.0,
                "warp_frequency": 0.06,
                "strikethrough_probability": 0.45,
                "strikethrough_width": 2,
                "strikethrough_color": "#202020",
                "random_seed": None,
            },
            "adv_docvqa": {
                "model_name": "donut",
                "checkpoint": "naver-clova-ix/donut-base-finetuned-docvqa",
                "local_files_only": True,
                "questions": None,
                "targets": None,
                "eps": 8.0,
                "steps": 20,
                "step_size": 1.0,
                "is_targeted": True,
                "mask": "include_all",
                "device": "cpu",
            },
        },
    }
    if not config_path.exists():
        return defaults
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    render_cfg = defaults["render"] | dict(user_cfg.get("render", {}))
    attack_cfg: Dict[str, Any] = {}
    user_attack = dict(user_cfg.get("attack", {}))
    for key, val in defaults["attack"].items():
        attack_cfg[key] = dict(val) | dict(user_attack.get(key, {}))
    return {"render": render_cfg, "attack": attack_cfg}

def build_render_config(cfg: Dict[str, Any]) -> RenderConfig:
    return RenderConfig(
        image_width=int(cfg.get("image_width", DEFAULT_RENDER_CONFIG.image_width)),
        image_height=int(cfg.get("image_height", DEFAULT_RENDER_CONFIG.image_height)),
        margin=int(cfg.get("margin", DEFAULT_RENDER_CONFIG.margin)),
        font_path=cfg.get("font_path", DEFAULT_RENDER_CONFIG.font_path),
        font_size=int(cfg.get("font_size", DEFAULT_RENDER_CONFIG.font_size)),
        dpi=int(cfg.get("dpi", DEFAULT_RENDER_CONFIG.dpi)),
        text_color=cfg.get("text_color", DEFAULT_RENDER_CONFIG.text_color),
        background_color=cfg.get("background_color", DEFAULT_RENDER_CONFIG.background_color),
    )

def build_attack_config(
    attack: str,
    *,
    attack_section: Dict[str, Any],
    random_seed: Optional[int],
) -> AttackConfig:
    semantic_cfg = None
    diacritics_cfg = None
    image_patch_cfg = None
    watermark_cfg = None
    distortions_cfg = None
    adv_docvqa_cfg = None
    if attack in ("semantic", "all"):
        sem = dict(attack_section.get("semantic", {}))
        semantic_cfg = SemanticAttackConfig(
            language=str(sem.get("language", "auto")),
            max_changed_words=int(sem.get("max_changed_words", 3)),
            population_size=int(sem.get("population_size", 24)),
            generations=int(sem.get("generations", 18)),
            tournament_k=int(sem.get("tournament_k", 3)),
            mutation_rate=float(sem.get("mutation_rate", 0.15)),
            crossover_rate=float(sem.get("crossover_rate", 0.7)),
            random_seed=random_seed if random_seed is not None else sem.get("random_seed"),
        )
    if attack in ("diacritics", "all"):
        dia = dict(attack_section.get("diacritics", {}))
        diacritics_cfg = DiacriticsAttackConfig(
            budget_per_word=int(dia.get("budget_per_word", 5)),
            diacritics_probability=float(dia.get("diacritics_probability", 0.6)),
            random_seed=random_seed if random_seed is not None else dia.get("random_seed"),
        )
    if attack in ("image_patch", "all"):
        ip = dict(attack_section.get("image_patch", {}))
        image_patch_cfg = ImagePatchAttackConfig(
            max_patches_per_line=int(ip.get("max_patches_per_line", 1)),
            patch_width_ratio=_tuple2(ip.get("patch_width_ratio"), (0.2, 0.6)),
            patch_height_ratio=_tuple2(ip.get("patch_height_ratio"), (0.2, 0.7)),
            pixel_value_min=int(ip.get("pixel_value_min", 40)),
            pixel_value_max=int(ip.get("pixel_value_max", 220)),
            pixel_fill_mode=str(ip.get("pixel_fill_mode", "random")),
            effects=_tuple_str(ip.get("effects"), ("bbox", "pixel", "text")),
            text_charset=str(ip.get("text_charset", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")),
            patch_text_length_range=(
                int(_tuple2(ip.get("patch_text_length_range"), (3, 8))[0]),
                int(_tuple2(ip.get("patch_text_length_range"), (3, 8))[1]),
            ),
            patch_font_size=ip.get("patch_font_size"),
            random_seed=random_seed if random_seed is not None else ip.get("random_seed"),
        )
    if attack in ("watermark", "all"):
        wm = dict(attack_section.get("watermark", {}))
        watermark_cfg = WatermarkAttackConfig(
            text_lines=tuple(wm.get("text_lines", ["CONFIDENTIAL"])),
            color=wm.get("color", "#606060"),
            alpha=int(wm.get("alpha", 80)),
            font_path=wm.get("font_path"),
            font_size=int(wm.get("font_size", 24)),
            x_spacing=int(wm.get("x_spacing", 160)),
            y_spacing=int(wm.get("y_spacing", 120)),
            angle_deg=float(wm.get("angle_deg", -18.0)),
            x_offset=int(wm.get("x_offset", 0)),
            y_offset=int(wm.get("y_offset", 0)),
        )
    if attack in ("distortions", "all"):
        dist = dict(attack_section.get("distortions", {}))
        distortions_cfg = DistortionsAttackConfig(
            enable_skew=bool(dist.get("enable_skew", True)),
            enable_rotate=bool(dist.get("enable_rotate", True)),
            enable_warp=bool(dist.get("enable_warp", True)),
            enable_strikethrough=bool(dist.get("enable_strikethrough", True)),
            character_distort_probability=float(dist.get("character_distort_probability", 0.18)),
            skew_degrees=float(dist.get("skew_degrees", 10.0)),
            rotate_degrees=float(dist.get("rotate_degrees", 9.0)),
            warp_probability=float(dist.get("warp_probability", 1.0)),
            warp_amplitude=float(dist.get("warp_amplitude", 2.0)),
            warp_frequency=float(dist.get("warp_frequency", 0.06)),
            strikethrough_probability=float(dist.get("strikethrough_probability", 0.45)),
            strikethrough_width=int(dist.get("strikethrough_width", 2)),
            strikethrough_color=dist.get("strikethrough_color", "#202020"),
            random_seed=random_seed if random_seed is not None else dist.get("random_seed"),
        )
    if attack in ("adv_docvqa", "all"):
        adv = dict(attack_section.get("adv_docvqa", {}))
        adv_docvqa_cfg = AdvDocVQAAttackConfig(
            model_name=str(adv.get("model_name", "donut")),
            checkpoint=adv.get("checkpoint", "naver-clova-ix/donut-base-finetuned-docvqa"),
            local_files_only=bool(adv.get("local_files_only", True)),
            questions=adv.get("questions"),
            targets=adv.get("targets"),
            eps=float(adv.get("eps", 8.0)),
            steps=int(adv.get("steps", 20)),
            step_size=float(adv.get("step_size", 1.0)),
            is_targeted=bool(adv.get("is_targeted", True)),
            mask=str(adv.get("mask", "include_all")),
            device=str(adv.get("device", "cpu")),
        )
    return AttackConfig(
        semantic=semantic_cfg,
        diacritics=diacritics_cfg,
        image_patch=image_patch_cfg,
        watermark=watermark_cfg,
        distortions=distortions_cfg,
        adv_docvqa=adv_docvqa_cfg,
    )

def init_def_subparser(def_subparser):
    def_subparser.add_argument("--config", default="config.json",
                        help="Путь к JSON-файлу конфигурации (по умолчанию config.json)")
    def_subparser.add_argument("--input", "-i", default="-",
                        help="Путь к файлу с текстом; '-' для чтения из stdin (по умолчанию)")
    def_subparser.add_argument("--output", "-o", default="output.png",
                        help="Путь к выходному изображению (по умолчанию output.png)")
    def_subparser.add_argument(
        "--attack", "-a",
        default="none",
        choices=["none", "semantic", "diacritics", "image_patch", "watermark", "distortions", "adv_docvqa", "all"],
        help="Тип защитных возмущений: semantic, diacritics, image_patch, watermark, distortions, adv_docvqa или all (комбинация)",
    )
    def_subparser.add_argument(
        "--random-seed", "-r",
        type=int,
        default=None,
        help="Фиксированный seed для воспроизводимости атак (опционально)",
    )
    def_subparser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Вывести в stderr итоговый текст после атак и краткие метаданные",
    )

def init_eval_subparser(eval_subparser):
    eval_subparser.add_argument("--config", default="config.json", help="Путь к JSON конфигурации рендера")
    eval_subparser.add_argument("--input", "-i", default="-", help="Файл с текстом или '-' для stdin")
    eval_subparser.add_argument("--output", "-o", default="ocr_results.json", help="Путь к JSON с результатами")
    eval_subparser.add_argument(
        "--engines", "-e",
        default=f'{",".join([k for k in ENGINE_RUNNERS.keys()])}',
        help=f"Список OCR-движков через запятую ({', '.join([k for k in ENGINE_RUNNERS.keys()])})",
    )
    eval_subparser.add_argument(
        "--attack", "-a",
        default="all",
        choices=["none", "semantic", "diacritics", "image_patch", "watermark", "distortions", "adv_docvqa", "all"],
        help="Тип(ы) возмущений",
    )
    eval_subparser.add_argument(
        "--dataset-mode",
        action="store_true",
        help="Оценивать не один текст, а набор текстов из --input (разделение по --separator).",
    )
    eval_subparser.add_argument(
        "--separator", "-s",
        default="\n---\n",
        help="Разделитель записей в dataset-mode.",
    )
    eval_subparser.add_argument(
        "--keep-empty",
        action="store_true",
        help="В dataset-mode не удалять пустые записи после split.",
    )
    eval_subparser.add_argument(
        "--print-samples",
        type=int,
        default=0,
        help="Сколько первых записей кратко вывести в stdout в dataset-mode (0 = не выводить).",
    )

def def_mode(args):
    # загрузка конфигурации
    config_path = Path(args.config)
    if args.verbose and not config_path.exists():
        print(f"Файл конфигурации '{config_path}' не найден, используются значения по умолчанию.", file=sys.stderr)
    cfg = load_app_config(config_path)
    render_config = build_render_config(cfg.get("render", {}))

    # чтение текста
    text = read_text(args.input)

    # рендеринг текста (с возмущениями или без)
    if args.attack == "none":
        with FreeTypeRenderer(render_config) as renderer:
            image, _ = renderer.render(text, x=0, y=0)
    else:
        attack_config = build_attack_config(
            args.attack,
            attack_section=cfg.get("attack", {}),
            random_seed=args.random_seed,
        )
        pipeline = AttackPipeline(render_config=render_config, attack_config=attack_config)
        image, attacked_text, meta = pipeline.render_attacked(text)
        if args.verbose:
            print(f"Итоговый текст после атак:\n{attacked_text}", file=sys.stderr)
            print(f"Метаданные атак: {meta}", file=sys.stderr)

    # сохранение результата
    image.save(args.output)
    if args.verbose:
        print(f"Изображение сохранено в {args.output}")

def eval_mode(args):
    config_path = Path(args.config)
    cfg = load_app_config(config_path)
    render_config = build_render_config(cfg.get("render", {}))
    attack_config = build_attack_config(
        args.attack,
        attack_section=cfg.get("attack", {}),
        random_seed=None,
    )
    pipeline = AttackPipeline(render_config=render_config, attack_config=attack_config)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    out_path = Path(args.output)

    if args.dataset_mode:
        texts = read_texts(args.input, args.separator, skip_empty=not args.keep_empty)
        results_list: List[Dict[str, Any]] = []
        for i, text in enumerate(texts):
            result = evaluate_ocr_engines(
                input_text=text,
                pipeline=pipeline,
                engines=engines,
                reference_text=text,
                output_path=None,
            )
            results_list.append(result)
            if i < args.print_samples:
                print(f"{i:03}: {text[:40]!r}")
        payload = {
            "dataset_mode": True,
            "separator": args.separator,
            "items_count": len(results_list),
            "items": results_list,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Результаты dataset-оценки сохранены в {out_path} (items={len(results_list)})")
        return

    text = read_text(args.input)
    results = evaluate_ocr_engines(
        input_text=text,
        pipeline=pipeline,
        engines=engines,
        reference_text=text,
        output_path=out_path,
    )

    print(f"Результаты сохранены в {out_path}")
    for engine, entry in results["metrics"].items():
        if entry.get("skipped"):
            print(f"{engine}: skipped ({entry.get('error')})")
        else:
            a = entry["attacked"]
            print(f"{engine}: attacked WER={a['wer']:.3f} CER={a['cer']:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Рендеринг текста в изображение с помощью FreeType")

    subparsers = parser.add_subparsers(help="Выбор режима работы утилиты", dest="command")
    def_subparser = subparsers.add_parser('def', help='Генерировать защищённое изображение текста')
    eval_subparser = subparsers.add_parser('eval', help='Тестировать качество защиты текста')

    init_def_subparser(def_subparser)
    init_eval_subparser(eval_subparser)

    args = parser.parse_args()

    if args.command == 'def':
        def_mode(args)
    elif args.command == 'eval':
        eval_mode(args)

if __name__ == '__main__':
    main()
