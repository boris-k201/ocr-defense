from .semantic import semantic_synonym_attack, SemanticAttackConfig
from .diacritics import diacritics_attack, DiacriticsAttackConfig
from .image_patch import image_patch_attack, ImagePatchAttackConfig
from .adv_docvqa_attack import AdvDocVQAAttackConfig, adv_docvqa_attack
from .watermark import WatermarkAttackConfig, watermark_attack
from .distortions import DistortionsAttackConfig, distortions_attack

__all__ = [
    "semantic_synonym_attack",
    "SemanticAttackConfig",
    "diacritics_attack",
    "DiacriticsAttackConfig",
    "image_patch_attack",
    "ImagePatchAttackConfig",
    "adv_docvqa_attack",
    "AdvDocVQAAttackConfig",
    "watermark_attack",
    "WatermarkAttackConfig",
    "distortions_attack",
    "DistortionsAttackConfig",
]

