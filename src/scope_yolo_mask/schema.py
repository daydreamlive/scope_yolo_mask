"""Configuration schema for YOLO mask pipeline."""

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, UsageType

# Common COCO class IDs for reference
COCO_CLASSES = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
}


class YOLOMaskConfig(BasePipelineConfig):
    """Configuration for YOLO26 segmentation preprocessor.

    This pipeline uses YOLO26-seg to detect and segment objects in video frames,
    outputting binary masks suitable for VACE conditioning.
    """

    pipeline_id = "yolo_mask"
    pipeline_name = "YOLO Mask"
    pipeline_description = (
        "Segments objects in video frames using YOLO26. "
        "Outputs binary masks for VACE inpainting/conditioning."
    )
    artifacts = []  # Ultralytics handles model downloads
    supports_prompts = False
    modified = True
    usage = [UsageType.PREPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}

    # Model configuration
    model_size: str = "nano"  # nano, small, medium, large, xlarge
    use_tensorrt: bool = False
    confidence_threshold: float = 0.5

    # Mask configuration
    target_classes: list[int] = [0]  # COCO class IDs (0 = person)
    invert_mask: bool = False  # Mask background instead of detected objects
