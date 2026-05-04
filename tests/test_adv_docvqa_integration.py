import unittest
from unittest.mock import patch

from PIL import Image

from ocr_defense.attacks.adv_docvqa_attack import AdvDocVQAAttackConfig
from ocr_defense.evaluation import AttackConfig, AttackPipeline
from ocr_defense.render import RenderConfig


class TestAdvDocVQAIntegration(unittest.TestCase):
    def test_pipeline_applies_adv_docvqa_attack(self):
        render_cfg = RenderConfig(image_width=320, image_height=120, font_size=20)
        attack_cfg = AttackConfig(
            adv_docvqa=AdvDocVQAAttackConfig(
                model_name="pix2struct",
                eps=4.0,
                steps=5,
                step_size=1.0,
            )
        )
        pipeline = AttackPipeline(render_config=render_cfg, attack_config=attack_cfg)

        fake_img = Image.new("RGB", (320, 120), (255, 255, 255))
        with patch("ocr_defense.evaluation.adv_docvqa_attack", return_value=(fake_img, {"ok": True})) as fn:
            img, attacked_text, meta = pipeline.render_attacked("Hello")

        self.assertIsInstance(img, Image.Image)
        self.assertEqual(attacked_text, "Hello")
        self.assertIn("adv_docvqa", meta)
        self.assertEqual(meta["adv_docvqa"]["ok"], True)
        fn.assert_called_once()


if __name__ == "__main__":
    unittest.main()

