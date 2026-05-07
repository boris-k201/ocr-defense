import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import json

# До импорта Paddle: отключить проверку доступности хостов моделей (шум и задержка).
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# Меньше предупреждений от tokenizers при многопоточности.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Тише прогресс-бары Hugging Face при загрузке весов.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from ocr_defense.attacks.diacritics import DiacriticsAttackConfig
from ocr_defense.attacks.adv_docvqa_attack import AdvDocVQAAttackConfig
from ocr_defense.attacks.image_patch import ImagePatchAttackConfig
from ocr_defense.attacks.semantic import SemanticAttackConfig
from ocr_defense.evaluation import AttackConfig, AttackPipeline, evaluate_ocr_engines
from ocr_defense.ocr_engines import ENGINE_RUNNERS
from ocr_defense.render import FreeTypeRenderer, load_render_config

def read_texts(text_path: str, separator: str) -> str:
    if text_path == "-":
        return sys.stdin.read().split(separator)
    input_path = Path(text_path)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    return input_path.read_text(encoding="utf-8").split(separator)

def main():
    parser = argparse.ArgumentParser(description="Проверка защиты текста в датасете")
    parser.add_argument("--config", default="config.json", help="Путь к JSON конфигурации рендера")
    parser.add_argument("--input", "-i", default="-", help="Файл датасета или '-' для stdin")
    parser.add_argument("--separator", "-s", default="\n---\n", help="Разделитель между записями в файле датасета")
    parser.add_argument("--output", "-o", default="ocr_results.json", help="Путь к JSON с результатами")
    parser.add_argument("--engines", "-e",
        default=f'{",".join([k for k in ENGINE_RUNNERS.keys()])}',
        help=f"Список OCR-движков через запятую ({', '.join([k for k in ENGINE_RUNNERS.keys()])})",
    )
    parser.add_argument("--attack", "-a",
        default="all",
        choices=["none", "semantic", "diacritics", "image_patch", "adv_docvqa", "all"],
        help="Тип(ы) возмущений",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    render_config = load_render_config(config_path)
    texts = read_texts(args.input, args.separator)

    semantic_cfg = None
    diacritics_cfg = None
    image_patch_cfg = None
    adv_docvqa_cfg = None

    if args.attack in ("semantic", "all"):
        semantic_cfg = SemanticAttackConfig(language="auto", max_changed_words=3, population_size=24, generations=18)
    if args.attack in ("diacritics", "all"):
        diacritics_cfg = DiacriticsAttackConfig(budget_per_word=5, diacritics_probability=0.6)
    if args.attack in ("image_patch", "all"):
        image_patch_cfg = ImagePatchAttackConfig(max_patches_per_line=1)
    if args.attack in ("adv_docvqa", "all"):
        adv_docvqa_cfg = AdvDocVQAAttackConfig(
            model_name="donut",
            checkpoint="naver-clova-ix/donut-base-finetuned-docvqa",
            local_files_only=True)

    attack_config = AttackConfig(
        semantic=semantic_cfg,
        diacritics=diacritics_cfg,
        image_patch=image_patch_cfg,
        adv_docvqa=adv_docvqa_cfg,
    )
    pipeline = AttackPipeline(render_config=render_config, attack_config=attack_config)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    out_path = Path(args.output)

    results = []
    for i, text in enumerate(texts):
        result = evaluate_ocr_engines(
            input_text=text,
            pipeline=pipeline,
            engines=engines,
            reference_text=text,
            output_path=None,
        )
        print(f"{i:02}: {text[:20]}..., {result['metrics']}")
        results.append(result)
    with open(args.output, 'wt') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены в {out_path}")

if __name__ == '__main__':
    main()
