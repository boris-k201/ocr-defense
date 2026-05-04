import unittest
from types import ModuleType
from unittest.mock import patch
import sys

from PIL import Image

from ocr_defense.attacks.adv_docvqa_attack import (
    AdvDocVQAAttackConfig,
    AdvDocVQANotAvailable,
    adv_docvqa_attack,
)


class TestAdvDocVQAAdapter(unittest.TestCase):
    def test_missing_dependency_raises(self):
        img = Image.new("RGB", (10, 10), (255, 255, 255))

        def fake_find_spec(name: str):
            if name == "secmlt":
                return None
            return object()

        with patch("importlib.util.find_spec", side_effect=fake_find_spec):
            with self.assertRaises(AdvDocVQANotAvailable):
                adv_docvqa_attack(img, AdvDocVQAAttackConfig())

    def test_wrapper_calls_pix2struct_attack(self):
        img = Image.new("RGB", (10, 10), (255, 255, 255))
        cfg = AdvDocVQAAttackConfig(
            model_name="pix2struct",
            questions=["Q1"],
            targets=["T1"],
            eps=4.0,
            steps=3,
            step_size=1.0,
            mask="include_all",
        )

        class DummyModel:  # noqa: D401
            """dummy"""

        class DummyProcessor:  # noqa: D401
            """dummy"""

        class DummyAutoProcessor:  # noqa: D401
            """dummy"""

        fake_mask = lambda t: t  # noqa: E731
        fake_out = Image.new("RGB", (10, 10), (1, 2, 3))

        m_attack = ModuleType("ocr_defense.attacks.adv_docvqa.attacks.pix2struct_attack")

        def fake_e2e_attack_pix2struct(**kwargs):
            return fake_out

        m_attack.e2e_attack_pix2struct = fake_e2e_attack_pix2struct

        m_models = ModuleType("ocr_defense.attacks.adv_docvqa.models.pix2struct")
        m_models.Pix2StructModel = lambda device="cpu": DummyModel()
        m_models.Pix2StructModelProcessor = lambda: DummyAutoProcessor()

        m_proc = ModuleType("ocr_defense.attacks.adv_docvqa.models.processing.pix2struct_processor")
        m_proc.Pix2StructImageProcessor = lambda: DummyProcessor()

        m_cfg = ModuleType("ocr_defense.attacks.adv_docvqa.config.config")
        m_cfg.AVAILABLE_MASKS = {"include_all": fake_mask}

        fake_modules = {
            "ocr_defense.attacks.adv_docvqa.attacks.pix2struct_attack": m_attack,
            "ocr_defense.attacks.adv_docvqa.models.pix2struct": m_models,
            "ocr_defense.attacks.adv_docvqa.models.processing.pix2struct_processor": m_proc,
            "ocr_defense.attacks.adv_docvqa.config.config": m_cfg,
        }

        with patch("importlib.util.find_spec", return_value=object()), patch.dict(sys.modules, fake_modules):
            out, meta = adv_docvqa_attack(img, cfg)

        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.getpixel((0, 0)), (1, 2, 3))
        self.assertEqual(meta["model_name"], "pix2struct")
        self.assertEqual(meta["questions_count"], 1)


if __name__ == "__main__":
    unittest.main()

