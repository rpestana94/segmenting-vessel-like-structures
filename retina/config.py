from __future__ import annotations
from dataclasses import dataclass, field

CROP_SIZE = 64           
BATCH_SIZE = 16         
CROPS_PER_IMAGE = 4      
BASE_FILTERS = 32       
DROPOUT = 0.1           
LEARNING_RATE = 1e-3      
INFER_SIDE = 576         
SR_SCALE = 3             
SR_LEARNING_RATE = 1e-3
GRAYSCALE_THRESHOLD = 100 
ROI_COVERAGE = 0.95
EROSION_ITER = 1
DILATION_ITER = 1
SEED = 20240501


@dataclass
class TrainConfig:
    kind: str = "unet"                
    epochs: int = 100
    steps_per_epoch: int = 100
    batch_size: int = BATCH_SIZE
    crop_size: int = CROP_SIZE
    crops_per_image: int = CROPS_PER_IMAGE
    base_filters: int = BASE_FILTERS
    dropout: float = DROPOUT
    learning_rate: float = LEARNING_RATE
    scale: int = SR_SCALE
    loss: str = "bce"                 
    augment: bool = True
    seed: int = SEED


@dataclass
class MaskConfig:
    threshold: int = GRAYSCALE_THRESHOLD
    roi_coverage: float = ROI_COVERAGE
    erosion_iter: int = EROSION_ITER
    dilation_iter: int = DILATION_ITER
    min_component_area: int = 0
    extra: dict = field(default_factory=dict)
