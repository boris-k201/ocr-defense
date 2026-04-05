from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image


class OCRNotAvailable(RuntimeError):
    pass


def _require_import(module_name: str):
    import importlib.util

    if importlib.util.find_spec(module_name) is None:
        raise OCRNotAvailable(f"Module '{module_name}' is not installed in the current venv.")


# Кэш тяжёлых моделей (evaluate вызывает OCR дважды: original + attacked).
_trocr_cache: Optional[Tuple[Any, Any]] = None
_easyocr_readers: Dict[Tuple[str, ...], Any] = {}
_paddleocr_instance: Optional[Any] = None


def ocr_tesseract(img: Image.Image, *, lang: str = "eng", psm: int = 6) -> str:
    _require_import("pytesseract")
    import pytesseract

    cfg = f"--psm {psm}"
    return pytesseract.image_to_string(img, lang=lang, config=cfg)


def ocr_easyocr(img: Image.Image, *, languages: Optional[list[str]] = None) -> str:
    _require_import("easyocr")
    import easyocr
    import numpy as np

    if languages is None:
        languages = ["en", "ru"]

    cache_key = tuple(languages)
    if cache_key not in _easyocr_readers:
        _easyocr_readers[cache_key] = easyocr.Reader(languages, gpu=False)
    reader = _easyocr_readers[cache_key]

    # EasyOCR expects RGB numpy array.
    rgb = img.convert("RGB")
    arr = np.asarray(rgb)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*pin_memory.*",
            category=UserWarning,
        )
        results = reader.readtext(arr, detail=1)
    # detail=1 -> list of (bbox, text, conf)
    texts = [r[1] for r in results]
    return "\n".join(texts)


def ocr_paddleocr(img: Image.Image, *, languages: Optional[list[str]] = None) -> str:
    # PaddleOCR зависит от пакета paddle (PaddlePaddle), а не только paddleocr.
    import importlib.util

    if importlib.util.find_spec("paddle") is None:
        raise OCRNotAvailable(
            "PaddlePaddle не установлен (import paddle). Установите: pip install paddlepaddle. "
            "Для CPU часто: pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/. "
            "Официальные колёса могут отсутствовать для очень новых версий Python — используйте 3.10–3.12."
        )

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    _require_import("paddleocr")
    from paddleocr import PaddleOCR
    import numpy as np

    if languages is None:
        languages = ["en", "ru"]

    # PaddleOCR accepts numpy arrays.
    arr = np.asarray(img.convert("RGB"))
    global _paddleocr_instance
    if _paddleocr_instance is None:
        _paddleocr_instance = PaddleOCR(use_angle_cls=True, lang="en")
    ocr = _paddleocr_instance
    # Note: multilingual config depends on PaddleOCR build; we keep a simple default.
    results = ocr.ocr(arr, cls=True)
    texts: list[str] = []
    for line in results:
        for _, (text, _conf) in line:
            texts.append(text)
    return "\n".join(texts)


def ocr_trocr(img: Image.Image) -> str:
    global _trocr_cache
    _require_import("transformers")
    _require_import("torch")
    from PIL import ImageOps
    import torch

    if _trocr_cache is None:
        # Меньше служебного вывода при загрузке весов.
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        model.eval()
        _trocr_cache = (processor, model)

    processor, model = _trocr_cache

    img_gray = ImageOps.grayscale(img)
    pixel_values = processor(images=img_gray, return_tensors="pt").pixel_values
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            generated_ids = model.generate(pixel_values, max_new_tokens=256)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


ENGINE_RUNNERS: Dict[str, Callable[[Image.Image], str]] = {
    "tesseract": ocr_tesseract,
    "easyocr": ocr_easyocr,
    "paddleocr": ocr_paddleocr,
    "trocr": ocr_trocr,
}

