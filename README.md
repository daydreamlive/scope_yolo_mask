# scope-yolo-mask

YOLO26 person segmentation preprocessor plugin for Daydream Scope.

## Features

- Real-time person detection and segmentation using YOLO26
- Binary mask output for VACE inpainting/conditioning
- Multiple model sizes (nano → xlarge)
- Optional TensorRT acceleration

## Installation

```bash
pip install -e .
```

Or with uv:

```bash
uv pip install -e .
```

## Usage

Once installed, the plugin automatically registers with Scope. The "YOLO Mask" pipeline will appear in the preprocessor list.

### Configuration

The pipeline can be configured with:

- `model_size`: "nano" (default), "small", "medium", "large", "xlarge"
- `use_tensorrt`: Enable TensorRT acceleration (default: False)
- `confidence_threshold`: Detection confidence (default: 0.5)

## How It Works

1. Receives video frames from upstream
2. Runs YOLO26-seg to detect people
3. Unions all person masks into a single binary mask per frame
4. Returns both:
   - `video`: Passthrough frames for display/downstream
   - `vace_input_masks`: Binary masks in VACE format `[1, 1, F, H, W]`

The masks can be used with VACE-enabled pipelines for person-aware generation.

## Model Storage

Models are downloaded to the Scope models directory:
- Default: `~/.daydream-scope/models/ultralytics/`
- Custom: Set `DAYDREAM_SCOPE_MODELS_DIR` environment variable

TensorRT engines are cached alongside the `.pt` files with `.engine` extension.

## Requirements

- ultralytics >= 8.4.0 (for YOLO26 support)
- torch
- Daydream Scope
