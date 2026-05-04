"""Scripts for preprocess data"""
from .pix2struct_processor import Pix2StructImageProcessor

try:  # optional, may fail on some transformers versions
    from .donut_processor import DonutImageProcessor
except Exception:  # pragma: no cover
    DonutImageProcessor = None  # type: ignore