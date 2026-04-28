import argparse
import os
import sys
from pathlib import Path
from typing import Optional

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
from ocr_defense.render import FreeTypeRenderer, load_render_config

def read_text(text_path: str) -> str:
    if text_path == "-":
        return sys.stdin.read()
    input_path = Path(text_path)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    return input_path.read_text(encoding="utf-8")

def build_attack_config(attack: str, *, random_seed: Optional[int]) -> AttackConfig:
    semantic_cfg = None
    diacritics_cfg = None
    image_patch_cfg = None
    if attack in ("semantic", "all"):
        semantic_cfg = SemanticAttackConfig(
            language="auto",
            max_changed_words=3,
            population_size=24,
            generations=18,
            random_seed=random_seed,
        )
    if attack in ("diacritics", "all"):
        diacritics_cfg = DiacriticsAttackConfig(
            budget_per_word=5,
            diacritics_probability=0.6,
            random_seed=random_seed,
        )
    if attack in ("image_patch", "all"):
        image_patch_cfg = ImagePatchAttackConfig(
            max_patches_per_line=1,
            random_seed=random_seed,
        )
    return AttackConfig(semantic=semantic_cfg, diacritics=diacritics_cfg, image_patch=image_patch_cfg)

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
        choices=["none", "semantic", "diacritics", "image_patch", "all"],
        help="Тип защитных возмущений: semantic, diacritics, image_patch или all (комбинация)",
    )
    def_subparser.add_argument(
        "--random-seed", "-r",
        type=int,
        default=None,
        help="Фиксированный seed для воспроизводимости атак (опционально)",
    )
    def_subparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Вывести в stderr итоговый текст после атак и краткие метаданные",
    )

def init_eval_subparser(eval_subparser):
    eval_subparser.add_argument("--config", default="config.json", help="Путь к JSON конфигурации рендера")
    eval_subparser.add_argument("--input", "-i", default="-", help="Файл с текстом или '-' для stdin")
    eval_subparser.add_argument("--output", "-o", default="ocr_results.json", help="Путь к JSON с результатами")
    eval_subparser.add_argument(
        "--engines", "-e",
        default="tesseract,trocr,easyocr",
        help="Список OCR-движков через запятую (tesseract, trocr, easyocr; paddleocr — только при установленном paddlepaddle)",
    )
    eval_subparser.add_argument(
        "--attack", "-a",
        default="all",
        choices=["none", "semantic", "diacritics", "image_patch", "all"],
        help="Тип(ы) возмущений",
    )

def def_mode(args):
    # загрузка конфигурации
    config_path = Path(args.config)
    if not config_path.exists():
        print(
            f"Файл конфигурации '{config_path}' не найден, используются значения по умолчанию.",
            file=sys.stderr,
        )
    render_config = load_render_config(config_path)

    # чтение текста
    if args.input == "-":
        text = sys.stdin.read()
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Файл с текстом '{input_path}' не найден.", file=sys.stderr)
            sys.exit(1)
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

    # рендеринг текста (с возмущениями или без)
    if args.attack == "none":
        with FreeTypeRenderer(render_config) as renderer:
            image = renderer.render(text, x=0, y=0, record_line_bboxes=False)
    else:
        attack_config = build_attack_config(args.attack, random_seed=args.random_seed)
        pipeline = AttackPipeline(render_config=render_config, attack_config=attack_config)
        image, attacked_text, meta = pipeline.render_attacked(text)
        if args.verbose:
            print(f"Итоговый текст после атак:\n{attacked_text}", file=sys.stderr)
            print(f"Метаданные атак: {meta}", file=sys.stderr)

    # сохранение результата
    image.save(args.output)
    print(f"Изображение сохранено в {args.output}")

def eval_mode(args):
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
