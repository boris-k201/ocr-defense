import base64
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image


class TestWebApp(unittest.TestCase):
    def test_pages_render(self):
        from webapp.app import app

        client = TestClient(app)
        r1 = client.get("/")
        r2 = client.get("/testing")
        self.assertEqual(r1.status_code, 200)
        self.assertIn("<!doctype html>", r1.text.lower())
        self.assertEqual(r2.status_code, 200)
        self.assertIn("<!doctype html>", r2.text.lower())

    def test_api_render_returns_data_url(self):
        from webapp import app as web

        client = TestClient(web.app)

        fake_img = Image.new("RGB", (20, 10), (255, 255, 255))
        fake_pipeline = MagicMock()
        fake_pipeline.render_original.return_value = (fake_img, None)

        with patch.object(web, "AttackPipeline", return_value=fake_pipeline):
            payload = {
                "text": "Hello",
                "attack": "none",
                "render": {
                    "image_width": 100,
                    "image_height": 50,
                    "margin": 10,
                    "font_path": None,
                    "font_size": 20,
                    "dpi": 96,
                    "text_color": "#000000",
                    "background_color": "#ffffff",
                },
                "advanced": {},
            }
            resp = client.post("/api/render", json=payload)

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("image_data_url", data)
        self.assertTrue(data["image_data_url"].startswith("data:image/png;base64,"))
        # Basic base64 sanity check
        b64 = data["image_data_url"].split(",", 1)[1]
        raw = base64.b64decode(b64)
        self.assertGreater(len(raw), 10)

    def test_api_evaluate_returns_metrics(self):
        from webapp import app as web

        client = TestClient(web.app)

        fake_results = {
            "reference_text": "X",
            "attacked_text": "X",
            "metrics": {"tesseract": {"skipped": False, "attacked": {"cer": 0.1, "wer": 0.2}}},
        }

        with patch.object(web, "evaluate_ocr_engines", return_value=fake_results):
            payload = {
                "text": "X",
                "engines": ["tesseract"],
                "attack": "none",
                "render": {
                    "image_width": 100,
                    "image_height": 50,
                    "margin": 10,
                    "font_path": None,
                    "font_size": 20,
                    "dpi": 96,
                    "text_color": "#000000",
                    "background_color": "#ffffff",
                },
                "advanced": {},
            }
            resp = client.post("/api/evaluate", json=payload)

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("metrics", data)
        self.assertIn("tesseract", data["metrics"])

    def test_build_attack_config_includes_adv_docvqa(self):
        from webapp import app as web

        adv = web.AdvancedAttackOptions(
            adv_docvqa=web.AdvDocVQAOptions(
                enabled=True,
                model_name="pix2struct",
                questions=["Q1"],
                targets=["T1"],
                eps=4.0,
                steps=5,
                step_size=1.0,
                is_targeted=True,
                mask="include_all",
                device="cpu",
            )
        )
        cfg = web._build_attack_config("adv_docvqa", adv)
        self.assertIsNotNone(cfg.adv_docvqa)
        self.assertEqual(cfg.adv_docvqa.model_name, "pix2struct")


if __name__ == "__main__":
    unittest.main()

