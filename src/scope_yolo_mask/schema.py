"""Configuration schema for YOLO mask pipeline."""

from typing import Literal

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)

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

TargetClass = Literal[
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
]


class YOLOMaskConfig(BasePipelineConfig):
    """Configuration for YOLO26 segmentation pipeline.

    This pipeline uses YOLO26-seg to detect and segment objects in video frames.
    Can be used as a preprocessor (outputting VACE masks) or standalone pipeline
    (outputting mask visualization or overlay).
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

    # Model configuration (load-time — changing these requires reloading)
    model_size: Literal["nano", "small", "medium", "large", "xlarge"] = Field(
        default="nano",
        description="YOLO model variant. Larger models are more accurate but slower.",
        json_schema_extra=ui_field_config(
            order=1, label="Model Size", is_load_param=True
        ),
    )
    # TODO: TensorRT support disabled for now — needs further testing for stability
    # use_tensorrt: bool = Field(
    #     default=False,
    #     description="Use TensorRT acceleration (requires TensorRT installed)",
    #     json_schema_extra=ui_field_config(
    #         order=2, label="TensorRT", is_load_param=True
    #     ),
    # )
    use_tensorrt: bool = False

    # Runtime configuration
    output_mode: Literal["mask", "overlay"] = Field(
        default="mask",
        description=(
            "Video output mode. "
            "'mask' shows the binary segmentation mask. "
            "'overlay' shows the mask blended on the original frame."
        ),
        json_schema_extra=ui_field_config(order=3, label="Output Mode"),
    )
    target_class: TargetClass = Field(
        default="person",
        description="Object class to detect and segment",
        json_schema_extra=ui_field_config(order=4, label="Target Class"),
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold",
        json_schema_extra=ui_field_config(order=5, label="Confidence"),
    )
    invert_mask: bool = Field(
        default=False,
        description="Invert the mask (segment background instead of detected objects)",
        json_schema_extra=ui_field_config(order=6, label="Invert Mask"),
    )
