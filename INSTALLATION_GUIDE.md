# GET_CAPTION Installation Guide

## Quick Setup

### 1. Clone and Setup Environment
```bash
git clone <repository-url>
cd GET_CAPTION
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Test Installation
```bash
# Place a test image in examples/
python examples/simple_example.py
```

## Detailed Installation

### System Requirements
- **Python**: 3.8+ (3.9+ recommended)
- **GPU**: CUDA-compatible GPU with 6GB+ VRAM (optional but recommended)
- **RAM**: 8GB+ system RAM
- **Storage**: 5GB+ for models and checkpoints

### Dependencies Breakdown

#### Core ML Libraries
```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install transformers>=4.20.0 tokenizers>=0.13.0
```

#### Computer Vision
```bash
pip install ultralytics>=8.0.0  # YOLOv8 and SAM2
pip install opencv-python>=4.5.0
pip install pillow>=9.0.0
```

#### Scientific Computing
```bash
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install matplotlib>=3.5.0
```

#### Utilities
```bash
pip install colorlog>=6.0.0
pip install tqdm>=4.60.0
pip install pyyaml>=6.0
```

### Model Downloads

Models are downloaded automatically on first use:

#### YOLOv8 Models (via ultralytics)
- `yolov8n.pt` (~6MB) - Fastest
- `yolov8s.pt` (~22MB) - Small  
- `yolov8m.pt` (~52MB) - Medium
- `yolov8l.pt` (~88MB) - Large

#### SAM2 Models (via ultralytics)
- `sam2_t.pt` (~39MB) - Tiny
- `sam2_s.pt` (~46MB) - Small
- `sam2_b.pt` (~159MB) - Base
- `sam2_l.pt` (~224MB) - Large

#### AlphaCLIP Models (included)
- Pre-downloaded in `AlphaCLIP/checkpoints/`
- ViT-B/32, ViT-B/16, ViT-L/14, RN50 variants

#### Language Models (via transformers)
- `bert-base-uncased` (~440MB)
- `roberta-base` (~500MB)
- Downloaded to `~/.cache/huggingface/`

### Verification

#### Test Core Components
```python
# Test AlphaCLIP
from src.captioning import AlphaCLIPWrapper
clip = AlphaCLIPWrapper("ViT-B/32", device="cpu")
print("✓ AlphaCLIP loaded")

# Test Detection
from src.detection import ObjectDetector  
detector = ObjectDetector("yolov8n.pt", device="cpu")
print("✓ YOLOv8 loaded")

# Test Language Model
from src.captioning import TextGenerator
generator = TextGenerator("bert-base-uncased", device="cpu") 
print("✓ BERT loaded")
```

#### Full Pipeline Test
```python
from src.main_pipeline import GetCaptionPipeline
from src.config import Config

config = Config()
config.model.device = "cpu"  # or "cuda"
pipeline = GetCaptionPipeline(config)
print("✓ Full pipeline initialized")
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce model sizes
config.model.detection_model = "yolov8n.pt"  # Instead of yolov8l.pt
config.model.alpha_clip_model = "ViT-B/32"   # Instead of ViT-L/14
config.processing.max_objects_per_image = 5  # Instead of 10
```

#### 2. Ultralytics Import Error
```bash
pip install ultralytics>=8.0.0
# Or for development version:
pip install git+https://github.com/ultralytics/ultralytics.git
```

#### 3. AlphaCLIP Import Error
- Ensure `AlphaCLIP/` directory exists in project root
- Check that `AlphaCLIP/alpha_clip/__init__.py` exists
- Verify Python path includes AlphaCLIP directory

#### 4. SAM2 Not Available
- Pipeline automatically falls back to bounding box masks
- No action needed, but segmentation quality will be reduced

#### 5. Transformers Model Download Issues
```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"
```

### Performance Optimization

#### GPU Setup
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

#### Memory Management
```python
# Enable memory efficient attention
import torch
torch.backends.cuda.enable_flash_sdp(True)

# Clear cache between batches  
torch.cuda.empty_cache()
```

#### Model Selection for Different Hardware

**Low-end (4GB GPU or CPU)**:
```python
config.model.detection_model = "yolov8n.pt"
config.model.sam_model = "sam2_t.pt" 
config.model.alpha_clip_model = "ViT-B/32"
config.model.lm_model = "bert-base-uncased"
config.processing.max_objects_per_image = 3
```

**Mid-range (8GB GPU)**:
```python
config.model.detection_model = "yolov8s.pt"
config.model.sam_model = "sam2_s.pt"
config.model.alpha_clip_model = "ViT-B/16" 
config.model.lm_model = "bert-base-uncased"
config.processing.max_objects_per_image = 5
```

**High-end (16GB+ GPU)**:
```python
config.model.detection_model = "yolov8l.pt"
config.model.sam_model = "sam2_l.pt"
config.model.alpha_clip_model = "ViT-L/14"
config.model.lm_model = "roberta-base"
config.processing.max_objects_per_image = 10
```

## Development Setup

### Additional Dev Dependencies
```bash
pip install pytest>=7.0.0 black>=22.0.0 flake8>=4.0.0
```

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black src/ examples/
flake8 src/ examples/
```

## Docker Setup (Optional)

```dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-m", "src.main_pipeline", "--help"]
```

## Support

If you encounter issues:
1. Check this guide for common solutions
2. Verify system requirements
3. Test individual components
4. Check GitHub issues
5. Create new issue with error details and system info
