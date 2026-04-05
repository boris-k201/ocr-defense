import argparse
import os
import sys
from pathlib import Path

# До импорта Paddle: отключить проверку доступности хостов моделей (шум и задержка).
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# Меньше предупреждений от tokenizers при многопоточности.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Тише прогресс-бары Hugging Face при загрузке весов.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from ocr_defense.attacks.diacritics import DiacriticsAttackConfig
from ocr_defense.attacks.image_patch import ImagePatchAttackConfig
from ocr_defense.attacks.semantic import SemanticAttackConfig
from ocr_defense.evaluation import AttackConfig, AttackPipeline, evaluate_ocr_engines
from ocr_defense.render import load_render_config


def read_text(text_path: str) -> str:
    if text_path == "-":
        return sys.stdin.read()
    input_path = Path(text_path)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    return input_path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR-устойчивый рендер текста + оценка OCR-движков")
    parser.add_argument("--config", default="config.json", help="Путь к JSON конфигурации рендера")
    parser.add_argument("--input", "-i", default="-", help="Файл с текстом или '-' для stdin")
    parser.add_argument("--output", "-o", default="ocr_results.json", help="Путь к JSON с результатами")
    parser.add_argument(
        "--engines",
        default="tesseract,trocr,easyocr",
        help="Список OCR-движков через запятую (tesseract, trocr, easyocr; paddleocr — только при установленном paddlepaddle)",
    )
    parser.add_argument(
        "--attack",
        default="all",
        choices=["none", "semantic", "diacritics", "image_patch", "all"],
        help="Тип(ы) возмущений",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    render_config = load_render_config(config_path)
    text = read_text(args.input)

    semantic_cfg = None
    diacritics_cfg = None
    image_patch_cfg = None

    if args.attack in ("semantic", "all"):
        semantic_cfg = SemanticAttackConfig(language="auto", max_changed_words=3, population_size=24, generations=18)
    if args.attack in ("diacritics", "all"):
        diacritics_cfg = DiacriticsAttackConfig(budget_per_word=5, diacritics_probability=0.6)
    if args.attack in ("image_patch", "all"):
        image_patch_cfg = ImagePatchAttackConfig(max_patches_per_line=1)

    attack_config = AttackConfig(semantic=semantic_cfg, diacritics=diacritics_cfg, image_patch=image_patch_cfg)
    pipeline = AttackPipeline(render_config=render_config, attack_config=attack_config)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    out_path = Path(args.output)

    results = evaluate_ocr_engines(
        input_text=text,
        pipeline=pipeline,
        engines=engines,
        reference_text=text,
        output_path=out_path,
    )

    print(f"Результаты сохранены в {out_path}")
    # Show a tiny summary to stdout.
    for engine, entry in results["metrics"].items():
        if entry.get("skipped"):
            print(f"{engine}: skipped ({entry.get('error')})")
        else:
            a = entry["attacked"]
            print(f"{engine}: attacked WER={a['wer']:.3f} CER={a['cer']:.3f}")


if __name__ == "__main__":
    main()

