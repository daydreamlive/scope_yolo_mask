"""YOLO26 person segmentation pipeline for realtime mask extraction."""

import logging
from typing import TYPE_CHECKING

import torch

from scope.core.config import get_models_dir
from scope.core.pipelines.interface import Pipeline, Requirements
from scope.core.pipelines.process import normalize_frame_sizes

from .schema import YOLOMaskConfig

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


class YOLOMaskPipeline(Pipeline):
    """YOLO26 person segmentation preprocessor.

    Detects people in video frames and outputs binary masks suitable for
    VACE conditioning. Returns both video passthrough and masks.
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
        confidence_threshold: float = 0.5,
        target_classes: list[int] | None = None,
        invert_mask: bool = False,
        **kwargs,
    ):
        """Initialize the YOLO mask pipeline.

        Args:
            device: Target device (defaults to CUDA if available)
            dtype: Data type for inference (default: float16)
            model_size: Model variant - nano/small/medium/large/xlarge
            use_tensorrt: Whether to use TensorRT acceleration
            confidence_threshold: Detection confidence threshold
            target_classes: COCO class IDs to segment (default: [0] for person)
            invert_mask: If True, mask background instead of detected objects
            **kwargs: Extra params from pipeline manager (ignored)
        """
        from ultralytics import YOLO

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes if target_classes is not None else [0]
        self.invert_mask = invert_mask

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
                engine_path = base_model.export(format="engine", half=dtype == torch.float16)
            self.model = YOLO(str(engine_path))
            logger.info(f"Loaded TensorRT engine: {engine_path}")
        else:
            self.model = YOLO(str(model_path))

        logger.info(f"YOLO26 model loaded on {self.device}")

    def prepare(self, **kwargs) -> Requirements:
        # Use chunk size from downstream pipeline (passed via parameters)
        return Requirements(input_size=12)

    @torch.no_grad()
    def __call__(self, **kwargs) -> dict:
        """Segment people in video frames.

        Args:
            video: Input video frames as list of tensors (THWC format, [0, 255] range)

        Returns:
            Dict with:
                - video: Passthrough frames (THWC, [0, 1] range) for queue
                - vace_input_frames: Video in VACE format [B, C, F, H, W], [-1, 1], on CUDA
                - vace_input_masks: Binary masks [B, 1, F, H, W] on CUDA
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for YOLOMaskPipeline")

        # Normalize frame sizes
        video = normalize_frame_sizes(video)

        masks_list = []
        passthrough_frames = []
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
                conf=self.confidence_threshold,
                classes=self.target_classes,
                verbose=False,
            )

            # Extract and union person masks (keep on GPU)
            result = results[0]
            if result.masks is not None and len(result.masks.data) > 0:
                # masks.data is (num_objects, H, W) - already on GPU
                all_masks = result.masks.data.float()
                combined_mask = all_masks.max(dim=0).values  # (H, W)
                combined_mask = combined_mask.to(self.device)
            else:
                # No persons detected - empty mask on GPU
                combined_mask = torch.zeros((h, w), dtype=torch.float32, device=self.device)

            # Ensure mask matches frame dimensions (on GPU)
            if combined_mask.shape != (h, w):
                combined_mask = torch.nn.functional.interpolate(
                    combined_mask.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode="nearest",
                ).squeeze()

            # Binary threshold and optional inversion
            combined_mask = (combined_mask > 0.5).float()
            if self.invert_mask:
                combined_mask = 1.0 - combined_mask
            masks_list.append(combined_mask)

            # Move frame to GPU and normalize to [0, 1]
            frame_gpu = frame_squeezed.to(self.device).float()
            if frame_gpu.max() > 1.0:
                frame_gpu = frame_gpu / 255.0

            # Store for passthrough (will move to CPU in bulk at end)
            passthrough_frames.append(frame_gpu)

            # VACE format: apply mask to frame (all on GPU)
            # mask=1 means "inpaint this region", fill with gray (0.5)
            mask_expanded = combined_mask.unsqueeze(-1)  # [H, W, 1]
            masked_frame = torch.where(
                mask_expanded > 0.5,
                torch.tensor(0.5, device=self.device, dtype=frame_gpu.dtype),
                frame_gpu,
            )
            # Normalize to [-1, 1] for VAE
            vace_frame = masked_frame * 2.0 - 1.0  # [H, W, C] in [-1, 1]
            vace_frames.append(vace_frame)

        # Stack passthrough frames: (T, H, W, C) on CPU for queue
        video_out = torch.stack(passthrough_frames, dim=0).cpu()

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
