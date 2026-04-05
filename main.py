import argparse
import sys
from pathlib import Path
from typing import Optional

from ocr_defense.attacks.diacritics import DiacriticsAttackConfig
from ocr_defense.attacks.image_patch import ImagePatchAttackConfig
from ocr_defense.attacks.semantic import SemanticAttackConfig
from ocr_defense.evaluation import AttackConfig, AttackPipeline
from ocr_defense.render import FreeTypeRenderer, load_render_config


def build_attack_config(attack: str, *, random_seed: Optional[int]) -> AttackConfig:
    """Те же профили возмущений, что и в evaluate.py."""
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


def main():
    parser = argparse.ArgumentParser(description="Рендеринг текста в изображение с помощью FreeType")
    parser.add_argument("--config", default="config.json",
                        help="Путь к JSON-файлу конфигурации (по умолчанию config.json)")
    parser.add_argument("--input", "-i", default="-",
                        help="Путь к файлу с текстом; '-' для чтения из stdin (по умолчанию)")
    parser.add_argument("--output", "-o", default="output.png",
                        help="Путь к выходному изображению (по умолчанию output.png)")
    parser.add_argument(
        "--attack",
        default="none",
        choices=["none", "semantic", "diacritics", "image_patch", "all"],
        help="Тип защитных возмущений: semantic, diacritics, image_patch или all (комбинация)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Фиксированный seed для воспроизводимости атак (опционально)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Вывести в stderr итоговый текст после атак и краткие метаданные",
    )
    args = parser.parse_args()

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

if __name__ == '__main__':
    main()
