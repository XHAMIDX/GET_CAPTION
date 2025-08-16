"""Configuration management for GET_CAPTION project."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelConfig:
    """Model configuration settings."""
    # AlphaCLIP settings
    alpha_clip_model: str = "ViT-L/14"  # ViT-B/32, ViT-B/16, ViT-L/14, RN50
    
    # Language model settings
    lm_model: str = "bert-base-uncased"  # or roberta-base
    
    # Object detection settings
    detection_model: str = "yolov8n.pt"  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
    detection_conf: float = 0.25
    detection_iou: float = 0.45
    
    # SAM2 settings
    sam_model: str = "sam2_b.pt"  # sam2_t.pt, sam2_s.pt, sam2_b.pt, sam2_l.pt
    
    # Device settings
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


@dataclass
class GenerationConfig:
    """Text generation configuration."""
    # Generation parameters
    sentence_len: int = 8
    candidate_k: int = 50
    num_iterations: int = 15
    
    # Scoring weights
    alpha: float = 0.8  # weight for fluency (BERT quality)
    beta: float = 1.5   # weight for image-matching degree (CLIP)
    gamma: float = 0.3  # weight for controllable degree (sentiment/POS)
    
    # Temperature and sampling
    lm_temperature: float = 0.3
    
    # Generation order
    order: str = "shuffle"  # sequential, shuffle, span, random
    
    # Prompt settings
    prompt: str = "A detailed image showing "
    
    # Control settings
    run_type: str = "caption"  # caption, controllable
    control_type: str = "sentiment"  # sentiment, pos
    sentiment_type: str = "positive"  # positive, negative
    pos_type: List[List[str]] = field(default_factory=lambda: [
        ['DET'], ['ADJ', 'NOUN'], ['NOUN'], ['VERB'], ['ADV'], ['ADP'], 
        ['DET', 'NOUN'], ['NOUN'], ['VERB'], ['ADP'], ['DET', 'NOUN']
    ])


@dataclass
class ProcessingConfig:
    """Image processing configuration."""
    # Input/Output paths
    input_path: str = "examples/"
    output_path: str = "results/"
    
    # Processing settings
    batch_size: int = 1
    samples_num: int = 3
    
    # Object detection filtering
    min_object_area: int = 1000  # minimum pixel area for objects
    max_objects_per_image: int = 10
    
    # Mask processing
    mask_blur_radius: int = 2
    mask_threshold: float = 0.5


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Logging
    log_level: str = "INFO"
    save_intermediate: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure output directory exists
        os.makedirs(self.processing.output_path, exist_ok=True)
        os.makedirs("logs", exist_ok=True)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return default."""
    if config_path and os.path.exists(config_path):
        # TODO: Implement YAML/JSON config loading
        pass
    return get_default_config()
