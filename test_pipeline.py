"""Test script for YOLO mask pipeline - validates mask and video output."""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add the plugin to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scope_yolo_mask.pipeline import YOLOMaskPipeline


def create_test_frame(image_path: str = None, size: tuple = (512, 512)) -> torch.Tensor:
    """Create a test frame from image or solid color."""
    if image_path and Path(image_path).exists():
        img = Image.open(image_path).convert("RGB")
        img = img.resize(size)
        frame_np = np.array(img).astype(np.float32)  # [H, W, C] in [0, 255]
    else:
        # Create a solid color test frame
        frame_np = np.full((*size, 3), 128, dtype=np.float32)

    # Convert to expected format: (1, H, W, C) tensor
    frame_tensor = torch.from_numpy(frame_np).unsqueeze(0)
    return frame_tensor


def main():
    print("=" * 60)
    print("YOLO Mask Pipeline Test")
    print("=" * 60)

    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    try:
        pipeline = YOLOMaskPipeline(
            model_size="nano",
            confidence_threshold=0.5,
        )
        print("   Pipeline initialized successfully")
    except Exception as e:
        print(f"   ERROR initializing pipeline: {e}")
        return

    # Find a test image with a person
    project_root = Path(__file__).parent.parent.parent.parent
    test_images = [
        project_root / "frontend/public/assets/woman1.jpg",
        project_root / "frontend/public/assets/woman2.jpg",
        project_root / "frontend/public/assets/example.png",
    ]

    test_image = None
    for img_path in test_images:
        if img_path.exists():
            test_image = str(img_path)
            break

    print(f"\n2. Creating test frame...")
    if test_image:
        print(f"   Using image: {test_image}")
    else:
        print("   Using solid color (no test image found)")

    frame = create_test_frame(test_image)
    print(f"   Frame shape: {frame.shape}")
    print(f"   Frame dtype: {frame.dtype}")
    print(f"   Frame range: [{frame.min():.1f}, {frame.max():.1f}]")

    # Run pipeline
    print("\n3. Running pipeline...")
    try:
        output = pipeline(video=[frame])
    except Exception as e:
        print(f"   ERROR running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # Validate output structure
    print("\n4. Validating output structure...")
    if not isinstance(output, dict):
        print(f"   ERROR: Expected dict, got {type(output)}")
        return

    if "video" not in output:
        print("   ERROR: Missing 'video' key")
        return

    if "vace_input_masks" not in output:
        print("   ERROR: Missing 'vace_input_masks' key")
        return

    print("   Output keys: ", list(output.keys()))

    # Validate video output
    print("\n5. Validating video output...")
    video_out = output["video"]
    print(f"   Shape: {video_out.shape}")
    print(f"   Dtype: {video_out.dtype}")
    print(f"   Range: [{video_out.min():.4f}, {video_out.max():.4f}]")
    print(f"   Mean: {video_out.mean():.4f}")

    if video_out.min() < 0:
        print("   WARNING: Video has negative values!")
    if video_out.max() > 1.0:
        print("   WARNING: Video values exceed 1.0!")

    # Validate mask output
    print("\n6. Validating mask output...")
    mask_out = output["vace_input_masks"]
    print(f"   Shape: {mask_out.shape}")
    print(f"   Dtype: {mask_out.dtype}")
    print(f"   Range: [{mask_out.min():.4f}, {mask_out.max():.4f}]")
    print(f"   Mean: {mask_out.mean():.4f}")

    # Check if mask is binary
    unique_values = torch.unique(mask_out)
    print(f"   Unique values: {unique_values.tolist()}")

    is_binary = len(unique_values) <= 2 and all(v in [0.0, 1.0] for v in unique_values.tolist())
    if is_binary:
        print("   Mask is BINARY (correct)")
    else:
        print("   WARNING: Mask is NOT binary - has intermediate values!")

    # Check expected shape [1, 1, F, H, W]
    expected_dims = 5
    if len(mask_out.shape) != expected_dims:
        print(f"   WARNING: Expected {expected_dims} dims, got {len(mask_out.shape)}")

    if mask_out.shape[0] != 1 or mask_out.shape[1] != 1:
        print(f"   WARNING: Expected [1, 1, F, H, W], got {mask_out.shape}")

    # Calculate mask coverage
    mask_coverage = mask_out.mean().item() * 100
    print(f"\n7. Mask statistics:")
    print(f"   Coverage: {mask_coverage:.1f}% of frame is masked (person detected)")

    if mask_coverage == 0:
        print("   NOTE: No person detected in frame")
    elif mask_coverage > 90:
        print("   WARNING: Very high coverage - possible detection issue")

    # Save visualization
    print("\n8. Saving visualizations...")
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)

    # Save mask as image (move to CPU for numpy conversion)
    mask_2d = mask_out[0, 0, 0].cpu().numpy()  # First frame
    mask_img = Image.fromarray((mask_2d * 255).astype(np.uint8), mode="L")
    mask_path = output_dir / "mask.png"
    mask_img.save(mask_path)
    print(f"   Saved mask: {mask_path}")

    # Save passthrough video frame
    video_2d = video_out[0].numpy()  # First frame, (H, W, C)
    video_img = Image.fromarray((video_2d * 255).astype(np.uint8), mode="RGB")
    video_path = output_dir / "passthrough.png"
    video_img.save(video_path)
    print(f"   Saved passthrough: {video_path}")

    # Save overlay
    if test_image:
        original = Image.open(test_image).convert("RGB").resize((512, 512))
        mask_rgba = Image.new("RGBA", (512, 512), (255, 0, 0, 0))
        mask_alpha = Image.fromarray((mask_2d * 128).astype(np.uint8), mode="L")
        mask_rgba.putalpha(mask_alpha)
        overlay = Image.alpha_composite(original.convert("RGBA"), mask_rgba)
        overlay_path = output_dir / "overlay.png"
        overlay.save(overlay_path)
        print(f"   Saved overlay: {overlay_path}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
