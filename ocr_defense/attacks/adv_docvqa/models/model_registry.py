from transformers import AutoProcessor

from .pix2struct import Pix2StructModel, Pix2StructModelProcessor
from .processing.pix2struct_processor import Pix2StructImageProcessor
from ..attacks.pix2struct_attack import e2e_attack_pix2struct

from .donut import DonutModel, DonutModelProcessor
from .processing.donut_processor import DonutImageProcessor
from ..attacks.donut_attack import attack_donut
from ..config.config import MODEL_NAMES, AVAILABLE_MASKS

def get_model(name:str, args):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}. Available models: {MODEL_NAMES}")

    # Initialize the model, processor, and attack based on the model name
    if name == 'pix2struct':
        processor = Pix2StructImageProcessor()
        autoprocessor = Pix2StructModelProcessor()
        model = Pix2StructModel(device=args.device)
        attack = e2e_attack_pix2struct

    elif name == 'donut':
        processor = DonutImageProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", use_fast=True) # use_fast=True is deprecated
        autoprocessor = DonutModelProcessor()
        model = DonutModel(device=args.device)
        attack = attack_donut
    
    # Retrive the mask function ptr based on available masks
    if args.mask not in AVAILABLE_MASKS:
        raise ValueError(f"Unknown mask: {args.mask}. Available masks: {list(AVAILABLE_MASKS.keys())}")
    mask_function = AVAILABLE_MASKS[args.mask]

    return processor, autoprocessor, model, attack, mask_function
