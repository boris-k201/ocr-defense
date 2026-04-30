from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from PIL import Image


class OCRNotAvailable(RuntimeError):
    """Raised when an OCR engine is not available in the current environment."""


def _require_import(module_name: str) -> None:
    import importlib.util

    if importlib.util.find_spec(module_name) is None:
        raise OCRNotAvailable(f"Module '{module_name}' is not installed in the current venv.")


# Caches (evaluate calls engines twice: original + attacked).
_trocr_cache: Optional[Tuple[Any, Any, str]] = None  # (processor, model, device)
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

    arr = np.asarray(img.convert("RGB"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)
        results = reader.readtext(arr, detail=1)
    texts = [r[1] for r in results]
    return "\n".join(texts)


def _extract_paddleocr_text(results: Any) -> str:
    """
    PaddleOCR can return:
    - list[ list[ (bbox, (text, conf)) ] ]  (older)
    - list[ (bbox, (text, conf)) ]         (some configs)
    - list[dict] with keys like 'text'     (newer/structured)
    We normalize all to a joined string.
    """
    texts: list[str] = []
    if results is None:
        return ""

    def handle_item(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            if item.strip():
                texts.append(item.strip())
            return
        if isinstance(item, dict):
            t = item.get("text") or item.get("rec_text") or item.get("ocr_text")
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
            return
        if isinstance(item, (list, tuple)) and len(item) == 2:
            # (bbox, (text, conf)) or (bbox, text)
            maybe = item[1]
            if isinstance(maybe, (list, tuple)) and len(maybe) >= 1 and isinstance(maybe[0], str):
                texts.append(maybe[0])
                return
            if isinstance(maybe, str):
                texts.append(maybe)
                return

    if isinstance(results, (list, tuple)):
        for x in results:
            if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)) and len(x) > 0 and len(x[0]) == 2:
                # nested list of lines
                for y in x:
                    handle_item(y)
            else:
                handle_item(x)
    else:
        handle_item(results)

    return "\n".join([t for t in texts if isinstance(t, str) and t.strip()])


def ocr_paddleocr(img: Image.Image, *, languages: Optional[list[str]] = None) -> str:
    import importlib.util

    # PaddleOCR requires paddle (PaddlePaddle) + paddleocr.
    if importlib.util.find_spec("paddle") is None:
        raise OCRNotAvailable(
            "PaddlePaddle is not installed (import paddle). Install paddlepaddle for your Python version, "
            "then install paddleocr."
        )
    _require_import("paddleocr")

    from paddleocr import PaddleOCR
    import numpy as np

    if languages is None:
        languages = ["en", "ru"]

    # PaddleOCR language selection is not fully multilingual by list.
    lang = "ru" if any(l.lower().startswith("ru") for l in languages) else "en"
    # Reduce noisy/slow connectivity checks on first import.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    global _paddleocr_instance
    if _paddleocr_instance is None:
        _paddleocr_instance = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False, use_gpu=False)
    ocr = _paddleocr_instance

    arr = np.asarray(img.convert("RGB"))
    results = ocr.ocr(arr, cls=True)
    return _extract_paddleocr_text(results)


def ocr_trocr(img: Image.Image) -> str:
    global _trocr_cache
    _require_import("transformers")
    _require_import("torch")
    import torch
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ModuleNotFoundError as e:
        # Some transformers installs pull optional deps lazily (e.g. soundfile/librosa).
        raise OCRNotAvailable(f"TrOCR is not available due to missing optional dependency: {e}") from e

    if _trocr_cache is None:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        _trocr_cache = (processor, model, device)

    processor, model, device = _trocr_cache

    pixel_values = processor(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            generated_ids = model.generate(pixel_values, max_new_tokens=256)
    out = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return out[0] if out else ""


ENGINE_RUNNERS: Dict[str, Callable[[Image.Image], str]] = {
    "tesseract": ocr_tesseract,
    "easyocr": ocr_easyocr,
    "paddleocr": ocr_paddleocr,
    "trocr": ocr_trocr,
}

