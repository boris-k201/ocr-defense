import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import ocr_defense.ocr_engines as oe


class TestOCREngines(unittest.TestCase):
    def setUp(self) -> None:
        # Clear caches between tests.
        oe._trocr_cache = None
        oe._easyocr_readers.clear()
        oe._paddleocr_instance = None

    def test_tesseract_calls_pytesseract(self):
        img = Image.new("RGB", (10, 10), (255, 255, 255))
        with patch.object(oe, "_require_import") as req, patch("pytesseract.image_to_string", return_value="ok") as itos:
            out = oe.ocr_tesseract(img, lang="eng", psm=6)
        req.assert_called_with("pytesseract")
        itos.assert_called()
        self.assertEqual(out, "ok")

    def test_easyocr_reader_is_cached_and_texts_joined(self):
        img = Image.new("RGB", (10, 10), (255, 255, 255))
        fake_reader = MagicMock()
        fake_reader.readtext.return_value = [([[0, 0], [1, 0], [1, 1], [0, 1]], "A", 0.9), ([], "B", 0.8)]

        with patch.object(oe, "_require_import") as req, patch("easyocr.Reader", return_value=fake_reader) as reader_ctor:
            out1 = oe.ocr_easyocr(img, languages=["en"])
            out2 = oe.ocr_easyocr(img, languages=["en"])

        req.assert_called_with("easyocr")
        self.assertEqual(reader_ctor.call_count, 1)
        self.assertEqual(out1, "A\nB")
        self.assertEqual(out2, "A\nB")

    def test_extract_paddleocr_text_handles_common_formats(self):
        # nested format
        r1 = [[([[0, 0], [1, 0], [1, 1], [0, 1]], ("X", 0.99))]]
        # flat format
        r2 = [([[0, 0], [1, 0], [1, 1], [0, 1]], ("Y", 0.5))]
        # dict format
        r3 = [{"text": "Z"}]
        self.assertEqual(oe._extract_paddleocr_text(r1), "X")
        self.assertEqual(oe._extract_paddleocr_text(r2), "Y")
        self.assertEqual(oe._extract_paddleocr_text(r3), "Z")

    def test_paddleocr_requires_paddle(self):
        img = Image.new("RGB", (10, 10), (255, 255, 255))
        # Simulate missing paddle
        with patch("importlib.util.find_spec", side_effect=lambda name: None if name == "paddle" else object()):
            with self.assertRaises(oe.OCRNotAvailable):
                oe.ocr_paddleocr(img, languages=["en"])

    def test_paddleocr_parses_results_and_caches_instance(self):
        img = Image.new("RGB", (10, 10), (255, 255, 255))

        class FakePaddleOCR:
            def __init__(self, *args, **kwargs):
                pass

            def ocr(self, arr, cls=True):
                return [[([[0, 0], [1, 0], [1, 1], [0, 1]], ("HELLO", 0.9))]]

        def find_spec(name: str):
            # paddle + paddleocr are "installed"
            return object()

        with patch("importlib.util.find_spec", side_effect=find_spec), patch.object(oe, "_require_import") as req:
            with patch("paddleocr.PaddleOCR", FakePaddleOCR) as ctor:
                out1 = oe.ocr_paddleocr(img, languages=["en"])
                out2 = oe.ocr_paddleocr(img, languages=["en"])

        req.assert_called_with("paddleocr")
        self.assertEqual(out1, "HELLO")
        self.assertEqual(out2, "HELLO")
        # ctor is a class here; check cache by ensuring it isn't recreated.
        self.assertIsNotNone(oe._paddleocr_instance)

    def test_trocr_returns_decoded_text_and_is_cached(self):
        img = Image.new("RGB", (10, 10), (255, 255, 255))

        class FakeProcessor:
            def __call__(self, images, return_tensors="pt"):
                import torch

                return MagicMock(pixel_values=torch.zeros((1, 3, 2, 2)))

            def batch_decode(self, ids, skip_special_tokens=True):
                return ["OK"]

        class FakeModel:
            def eval(self):
                return self

            def to(self, device):
                return self

            def generate(self, pixel_values, max_new_tokens=256):
                import torch

                return torch.tensor([[1, 2, 3]])

        def find_spec(name: str):
            return object()

        # Provide a fake 'transformers' module so the test doesn't pull real optional deps.
        class FakeTrOCRProcessor:
            @staticmethod
            def from_pretrained(name: str):
                return FakeProcessor()

        class FakeVisionEncoderDecoderModel:
            @staticmethod
            def from_pretrained(name: str):
                return FakeModel()

        fake_transformers = MagicMock(
            TrOCRProcessor=FakeTrOCRProcessor,
            VisionEncoderDecoderModel=FakeVisionEncoderDecoderModel,
        )

        with patch("importlib.util.find_spec", side_effect=find_spec), patch.dict("sys.modules", {"transformers": fake_transformers}):
            out1 = oe.ocr_trocr(img)
            out2 = oe.ocr_trocr(img)

        self.assertEqual(out1, "OK")
        self.assertEqual(out2, "OK")
        # cached
        self.assertIsNotNone(oe._trocr_cache)


if __name__ == "__main__":
    unittest.main()

