"""YOLO26 segmentation pipeline for realtime mask extraction."""

import logging
from typing import TYPE_CHECKING

import torch

from scope.core.config import get_models_dir
from scope.core.pipelines.interface import Pipeline, Requirements
from scope.core.pipelines.process import normalize_frame_sizes

from .schema import COCO_CLASSES, YOLOMaskConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)

# Model size variants
MODEL_VARIANTS = {
    "nano": "yolo26n-seg.pt",
    "small": "yolo26s-seg.pt",
    "medium": "yolo26m-seg.pt",
    "large": "yolo26l-seg.pt",
    "xlarge": "yolo26x-seg.pt",
}

# Subdirectory for YOLO models within scope models dir
YOLO_MODELS_SUBDIR = "ultralytics"

# Overlay color (green) and blend alpha
OVERLAY_COLOR = (0.0, 0.8, 0.0)
OVERLAY_ALPHA = 0.4


class YOLOMaskPipeline(Pipeline):
    """YOLO26 segmentation pipeline.

    Detects and segments objects in video frames. Can be used as:
    - Preprocessor: outputs VACE-compatible masks for downstream pipelines
    - Standalone pipeline: outputs mask visualization or overlay
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return YOLOMaskConfig

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        model_size: str = "nano",
        use_tensorrt: bool = False,
        **kwargs,
    ):
        """Initialize the YOLO mask pipeline.

        Args:
            device: Target device (defaults to CUDA if available)
            dtype: Data type for inference (default: float16)
            model_size: Model variant - nano/small/medium/large/xlarge
            use_tensorrt: Whether to use TensorRT acceleration
            **kwargs: Extra params from pipeline manager (ignored)
        """
        from ultralytics import YOLO

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        # Get model path in scope models directory
        model_filename = MODEL_VARIANTS.get(model_size, MODEL_VARIANTS["nano"])
        models_dir = get_models_dir() / YOLO_MODELS_SUBDIR
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / model_filename

        logger.info(f"Loading YOLO26 segmentation model: {model_path}")

        if use_tensorrt:
            # Export to TensorRT if needed, then load engine
            engine_path = model_path.with_suffix(".engine")
            if not engine_path.exists():
                base_model = YOLO(str(model_path))
                engine_path = base_model.export(
                    format="engine", half=dtype == torch.float16
                )
            self.model = YOLO(str(engine_path))
            logger.info(f"Loaded TensorRT engine: {engine_path}")
        else:
            self.model = YOLO(str(model_path))

        logger.info(f"YOLO26 model loaded on {self.device}")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=12)

    @torch.no_grad()
    def __call__(self, **kwargs) -> dict:
        """Segment objects in video frames.

        Args:
            video: Input video frames as list of tensors (THWC format, [0, 255] range)
            output_mode: "mask" for binary mask, "overlay" for mask on original frame
            target_class: COCO class name to segment (e.g. "person", "car")
            confidence_threshold: Detection confidence threshold
            invert_mask: If True, invert the mask

        Returns:
            Dict with:
                - video: Output frames (THWC, [0, 1] range) based on output_mode
                - vace_input_frames: Video in VACE format [B, C, F, H, W], [-1, 1]
                - vace_input_masks: Binary masks [B, 1, F, H, W]
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for YOLOMaskPipeline")

        # Runtime parameters
        output_mode = kwargs.get("output_mode", "mask")
        target_class = kwargs.get("target_class", "person")
        confidence_threshold = kwargs.get("confidence_threshold", 0.5)
        invert_mask = kwargs.get("invert_mask", False)

        # Convert class name to COCO class ID
        target_class_id = COCO_CLASSES.get(target_class, 0)

        # Normalize frame sizes
        video = normalize_frame_sizes(video)

        masks_list = []
        display_frames = []
        vace_frames = []

        for frame in video:
            # frame is (1, H, W, C) tensor
            frame_squeezed = frame.squeeze(0)  # (H, W, C)

            # CPU transfer for YOLO input (required - YOLO expects numpy)
            frame_np = frame_squeezed.cpu().numpy()
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype("uint8")
            else:
                frame_np = frame_np.astype("uint8")

            h, w = frame_np.shape[:2]

            # Run YOLO inference
            results = self.model(
                frame_np,
                conf=confidence_threshold,
                classes=[target_class_id],
                verbose=False,
            )

            # Extract and union masks (keep on GPU)
            result = results[0]
            if result.masks is not None and len(result.masks.data) > 0:
                # masks.data is (num_objects, H, W) - already on GPU
                all_masks = result.masks.data.float()
                combined_mask = all_masks.max(dim=0).values  # (H, W)
                combined_mask = combined_mask.to(self.device)
            else:
                # No objects detected - empty mask on GPU
                combined_mask = torch.zeros(
                    (h, w), dtype=torch.float32, device=self.device
                )

            # Ensure mask matches frame dimensions (on GPU)
            if combined_mask.shape != (h, w):
                combined_mask = torch.nn.functional.interpolate(
                    combined_mask.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode="nearest",
                ).squeeze()

            # Binary threshold and optional inversion
            combined_mask = (combined_mask > 0.5).float()
            if invert_mask:
                combined_mask = 1.0 - combined_mask
            masks_list.append(combined_mask)

            # Move frame to GPU and normalize to [0, 1]
            frame_gpu = frame_squeezed.to(self.device).float()
            if frame_gpu.max() > 1.0:
                frame_gpu = frame_gpu / 255.0

            # Build display frame based on output mode
            mask_expanded = combined_mask.unsqueeze(-1)  # [H, W, 1]
            if output_mode == "overlay":
                # Blend colored mask with original frame
                overlay_color = torch.tensor(
                    OVERLAY_COLOR, device=self.device, dtype=frame_gpu.dtype
                )
                color_layer = mask_expanded * overlay_color  # [H, W, 3]
                display_frame = (
                    frame_gpu * (1.0 - mask_expanded * OVERLAY_ALPHA)
                    + color_layer * OVERLAY_ALPHA
                )
            else:
                # Binary mask as 3-channel grayscale image
                display_frame = mask_expanded.expand_as(frame_gpu)

            display_frames.append(display_frame)

            # VACE format: apply mask to frame (all on GPU)
            # mask=1 means "inpaint this region", fill with gray (0.5)
            masked_frame = torch.where(
                mask_expanded > 0.5,
                torch.tensor(0.5, device=self.device, dtype=frame_gpu.dtype),
                frame_gpu,
            )
            # Normalize to [-1, 1] for VAE
            vace_frame = masked_frame * 2.0 - 1.0  # [H, W, C] in [-1, 1]
            vace_frames.append(vace_frame)

        # Stack display frames: (T, H, W, C) on CPU for queue
        video_out = torch.stack(display_frames, dim=0).cpu()

        # Stack VACE frames: [F, H, W, C] -> [1, C, F, H, W] on CUDA
        vace_video = torch.stack(vace_frames, dim=0)  # (F, H, W, C)
        vace_video = vace_video.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, F, H, W)

        # Stack masks: [F, H, W] -> [1, 1, F, H, W] on CUDA
        masks_tensor = torch.stack(masks_list, dim=0)  # (F, H, W)
        vace_masks = masks_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, F, H, W)

        return {
            "video": video_out,
            "vace_input_frames": vace_video,
            "vace_input_masks": vace_masks,
        }
