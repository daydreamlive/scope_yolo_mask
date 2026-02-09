# scope-yolo-mask

A Scope plugin that segments objects in real-time video using YOLO26, producing binary masks for VACE inpainting and conditioning.

## Features

- **Object Segmentation** — Real-time detection and segmentation of 19 COCO object classes (person, car, dog, etc.) using YOLO26-seg (video mode)
- **Mask Output** — Binary segmentation mask displayed as a grayscale image, suitable for debugging and mask preview
- **Overlay Output** — Mask blended on the original frame with a green tint for intuitive visualization
- **Invert Mask** — Option to invert the mask to segment the background instead of the detected object
- **VACE Integration** — Outputs masks in VACE format (`[1, 1, F, H, W]`) for downstream inpainting pipelines
- **Multiple Model Sizes** — Choose from nano, small, medium, large, or xlarge for speed vs. accuracy tradeoffs

## Install

Follow the [Scope plugins guide](https://github.com/daydreamlive/scope/blob/main/docs/plugins.md) to install this plugin using the URL:

```
https://github.com/daydreamlive/scope_yolo_mask.git
```

## Upgrade

Follow the [Scope plugins guide](https://github.com/daydreamlive/scope/blob/main/docs/plugins.md) to upgrade this plugin to the latest version.

## Architecture

This plugin registers one pipeline via the `register_pipelines` hook in `plugin.py`:

### YOLO Mask (`yolo_mask`)

A **video-mode** preprocessor pipeline that segments objects in each input frame using YOLO26-seg. It accepts a `model_size` load parameter (`nano`/`small`/`medium`/`large`/`xlarge`, default `nano`) that determines the YOLO model variant. At runtime, `target_class` selects which COCO class to detect (default `person`), `confidence_threshold` (0.0–1.0, default 0.5) filters low-confidence detections, `output_mode` (`mask` or `overlay`) controls the display output, and `invert_mask` flips the segmented region.

The pipeline outputs three tensors:
- `video` — display frames `(T, H, W, 3)` in `[0, 1]` range (mask visualization or overlay)
- `vace_input_frames` — masked video in VACE format `[1, C, F, H, W]` in `[-1, 1]` range
- `vace_input_masks` — binary masks `[1, 1, F, H, W]` where 1 indicates regions to inpaint

Models are downloaded automatically by Ultralytics to `~/.daydream-scope/models/ultralytics/`.
