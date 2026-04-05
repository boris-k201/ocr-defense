import unittest

from ocr_defense.attacks.diacritics import DiacriticsAttackConfig
from ocr_defense.evaluation import AttackConfig, AttackPipeline, evaluate_ocr_engines
from ocr_defense.render import RenderConfig


class TestEvaluationSkips(unittest.TestCase):
    def test_evaluate_marks_skipped_when_engine_missing(self):
        render_cfg = RenderConfig(image_width=240, image_height=100, font_size=18)
        pipeline = AttackPipeline(
            render_config=render_cfg,
            attack_config=AttackConfig(
                diacritics=DiacriticsAttackConfig(budget_per_word=2, diacritics_probability=1.0, random_seed=0)
            ),
        )
        out = evaluate_ocr_engines(
            input_text="Hello OCR",
            pipeline=pipeline,
            # Use an unknown engine name to avoid depending on installed OCR packages.
            engines=("unknown_engine_name_123",),
        )

        self.assertIn("unknown_engine_name_123", out["metrics"])
        self.assertTrue(out["metrics"]["unknown_engine_name_123"]["skipped"])


if __name__ == "__main__":
    unittest.main()

