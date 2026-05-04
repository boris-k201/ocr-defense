from ..attacks.masks import mask_include_all, mask_exclude_white, mask_bottom_right_corner

MODEL_NAMES = ['pix2struct','donut']

AVAILABLE_MASKS = {
    "include_all": mask_include_all,
    "exclude_white": mask_exclude_white,
    "bottom_right_corner": mask_bottom_right_corner
}