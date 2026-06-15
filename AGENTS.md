# AGENTS.md - OCEC Project Guidelines for AI Agents

This file contains guidelines for AI agents working on the OCEC (Open Closed Eyes Classification) codebase.

## Build and Execution Commands

### Dependency Management
```bash
# Install dependencies using uv (Python 3.11.12)
uv sync
source .venv/bin/activate

# Install a specific dependency
uv add <package_name>
```

### Running the Code
```bash
# Training entry point
uv run python -m ocec train --data_root data/dataset.parquet --output_dir runs/ocec

# Demo/inference
uv run python demo_ocec.py -v 0 -m model.onnx -om ocec_l.onnx -ep cuda

# Dataset preparation
uv run python 01_dataset_viewer.py --split train
uv run python 03_wholebody34_data_extractor.py -ea -m model.onnx -oep tensorrt
uv run python 04_dataset_convert_to_parquet.py --annotation data/cropped/annotation.csv --output data/dataset.parquet

# ONNX export
uv run python -m ocec exportonnx --checkpoint runs/ocec/best.pt --output ocec.onnx --opset 17
```

### Testing
- This project does not use formal unit testing frameworks (pytest/unittest)
- Manual testing via demo scripts and training runs
- Verify models by running inference on test videos and checking output
- Test dataset splits: train (80%), val (20%), test (optional)

### Training Scripts
```bash
# Example training with shell scripts
bash train.sh
bash train_convnext.sh

# TensorBoard monitoring
tensorboard --logdir runs/ocec
```

## Code Style Guidelines

### Import Organization
```python
# 1. Future imports (always first)
from __future__ import annotations

# 2. Standard library imports
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# 3. Third-party imports
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

# 4. Local imports
from .model import OCEC
from .data import OCECDataset
```

### Type Hints
- Use `from __future__ import annotations` for forward references
- All functions should have type hints for parameters and returns
- Use `Optional[T]` for nullable types
- Use `Sequence[T]` instead of `List[T]` for read-only collections
- Private internal classes start with underscore: `_SepConvBlock`, `_InvertedResidualSEBlock`

### Naming Conventions
- **Classes**: PascalCase - `OCECDataset`, `ModelConfig`, `ArcFaceHead`
- **Functions**: snake_case - `collect_samples()`, `create_dataloader()`
- **Constants**: UPPER_CASE - `DEFAULT_MEAN`, `CONF_THRESHOLD`
- **Private methods**: _snake_case - `_init_weights()`, `_make_block()`
- **Variables**: snake_case - `image_size`, `batch_size`
- **Parameters**: descriptive names - `train_ratio`, `val_ratio`, `test_ratio`

### File Structure
- `ocec/__init__.py` - Package exports
- `ocec/model.py` - Neural network architectures
- `ocec/data.py` - Dataset classes and data loading
- `ocec/pipeline.py` - Training pipeline and CLI
- Root scripts numbered: `01_dataset_viewer.py`, `02_real_data_size_hist.py`, etc.

### Formatting
- Indentation: 4 spaces
- Line length: no strict limit, aim for readability
- Imports: group as shown above, blank line between groups
- Blank lines: 2 blank lines between top-level functions, 1 blank line in classes
- Trailing whitespace: avoid

### Error Handling
```python
try:
    # Operation that may fail
    image = Image.open(io.BytesIO(sample.image_bytes)).convert("RGB")
except Exception as e:
    logging.warning(f"Failed to load image: {e}")
    # Handle gracefully: skip, return None, raise custom exception
```

### Logging
- Use Python's `logging` module, not `print()`
- Logger: `logger = logging.getLogger(__name__)`
- Log levels: DEBUG for detailed info, INFO for normal operations, WARNING for issues, ERROR for failures
- Format: `logger.info("Message with %s placeholder", variable)`

### Configuration
- Use `@dataclass` for configuration objects
- Config parameters include: `base_channels`, `num_blocks`, `arch_variant`, `head_variant`, `dropout`, `image_size`
- Default values in dataclass definitions

### PyTorch Specific
- Model class inherits from `nn.Module`
- Forward pass signature: `def forward(self, x: torch.Tensor, labels=None) -> torch.Tensor:`
- Use `inplace=True` for activations when possible: `nn.ReLU(inplace=True)`
- BatchNorm after convolutions
- Weight initialization: Kaiming for Conv, Xavier for Linear, ones for BN weights

### Data Loading
- Dataset class inherits from `Dataset` and implements `__len__()` and `__getitem__()`
- DataLoader with `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=8`
- Use `WeightedRandomSampler` for imbalanced datasets
- Support both disk loading and embedded image bytes
- Pre-load images to RAM in dataset `__init__()` for performance

### ONNX Export
- Export with dynamic batch dimension
- Simplify model with `onnxsim` after export
- Rewrite BatchNorm to Mul/Add primitives
- Input: `images`, Output: `prob_open`

### Chinese Comments
- Mixed Chinese and English comments in code
- Maintain existing language when editing
- Use clear descriptions in either language

### Paths
- Always use `pathlib.Path` instead of string paths
- Resolve paths with `.resolve()`
- Check existence with `.exists()`
- Use `/` operator for path joining: `data_dir / path_candidate`

### Architecture Variants
Three supported architectures in `ocec/model.py`:
- `baseline`: Depthwise separable CNN with avg pooling
- `inverted_se`: MobileNetV2-style with SE attention, avgmax_mlp head
- `convnext`: ConvNeXt blocks with transformer head, requires divisible feature map

### Training Configuration
- Loss: `BCEWithLogitsLoss` with `pos_weight`
- Optimizer: Adam or AdamW
- Mixed precision: `--use_amp` flag
- Checkpoints: `ocec_epoch_*.pt` and `ocec_best_epochXXXX_f1_YYYY.pt`
- Resume: `--resume <checkpoint>` restores optimizer/scheduler/AMP states
- Metrics: logged to TensorBoard

### No Linting Tools Configured
- This project does not currently use black, ruff, pylint, mypy, or similar tools
- Follow style conventions outlined here
- Maintain consistency with existing code patterns
