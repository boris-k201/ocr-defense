import unittest

from ocr_defense.attacks.diacritics import DiacriticsAttackConfig, diacritics_attack
from ocr_defense.attacks.image_patch import ImagePatchAttackConfig, image_patch_attack
from ocr_defense.attacks.semantic import SemanticAttackConfig, semantic_synonym_attack
from ocr_defense.render import FreeTypeRenderer, RenderConfig


class TestAttacks(unittest.TestCase):
    def test_diacritics_budget_per_word(self):
        text = "Hello world"
        cfg = DiacriticsAttackConfig(budget_per_word=5, diacritics_probability=1.0, random_seed=123)
        attacked, meta = diacritics_attack(text, cfg)
        # Check each whitespace token that contains letters.
        for token in attacked.split():
            letters = any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in token)
            if not letters:
                continue
            diacs = sum(1 for ch in token if 0x0300 <= ord(ch) <= 0x036F)
            self.assertLessEqual(diacs, cfg.budget_per_word)
        self.assertNotEqual(attacked, text)
        self.assertIn("marks_total", meta)

    def test_semantic_synonym_attack_changes_within_budget(self):
        text = "good method"
        cfg = SemanticAttackConfig(
            language="en",
            max_changed_words=1,
            population_size=12,
            generations=6,
            random_seed=42,
        )
        attacked, meta = semantic_synonym_attack(text, cfg)
        self.assertLessEqual(meta["changed_words"], cfg.max_changed_words)
        self.assertNotEqual(attacked, text)

    def test_image_patch_attack_modifies_image(self):
        render_cfg = RenderConfig(image_width=320, image_height=120, font_size=22)
        with FreeTypeRenderer(render_cfg) as renderer:
            text = "Hi"
            img, bboxes = renderer.render(text, x=0, y=0, record_line_bboxes=True)

            patch_cfg = ImagePatchAttackConfig(
                max_patches_per_line=1,
                effects=("pixel",),
                random_seed=1,
                pixel_fill_mode="random",
            )
            attacked = image_patch_attack(img, renderer=renderer, line_bboxes=bboxes, config=patch_cfg)

        # Ensure something changed.
        diff = sum(
            1
            for y in range(img.height)
            for x in range(img.width)
            if img.getpixel((x, y)) != attacked.getpixel((x, y))
        )
        self.assertGreater(diff, 0)


if __name__ == "__main__":
    unittest.main()

