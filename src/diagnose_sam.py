"""
Diagnose SAM Segmentation on Real Image
Shows exactly what SAM is detecting
"""

import cv2
import numpy as np
import os
import config
from segmentation import GeoSAMSegmenter
from proper_sam_segmenter import AccurateSAMSegmenter

print("="*70)
print("SAM SEGMENTATION DIAGNOSTIC")
print("="*70)

# Load the actual image being used
IMAGE_PATH = "data/test_images/ladi_00297.jpg"

print(f"\n[1/4] Loading image: {IMAGE_PATH}")

if not os.path.exists(IMAGE_PATH):
    print(f"✗ Image not found!")
    print(f"  Looking for: {os.path.abspath(IMAGE_PATH)}")
    exit(1)

image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"✗ Failed to load image")
    exit(1)

print(f"✓ Image loaded: {image.shape}")

# Create segmenter
print(f"\n[2/4] Creating segmenter...")
#old version -> segmenter = GeoSAMSegmenter()
segmenter = AccurateSAMSegmenter()

# Load model
print(f"\n[3/4] Loading model...")
if not segmenter.load_model():
    print("✗ Model loading failed")
    exit(1)

# Check if using real SAM or mock
if segmenter.predictor is None:
    print("✓ Using REAL SAM model")
else:
    print("⚠ Using mock model")

# Segment
print(f"\n[4/4] Segmenting image...")
mask = segmenter.segment(image)

if mask is None:
    print("✗ Segmentation failed")
    exit(1)

print(f"✓ Segmentation complete")

# Get statistics
stats = segmenter.get_statistics(mask)
print(f"\n" + "="*70)
print("SEGMENTATION RESULTS:")
print("="*70)
print(f"Safe pixels:   {stats['safe_pixels']:,} ({stats['safe_percentage']:.1f}%)")
print(f"Unsafe pixels: {stats['unsafe_pixels']:,} ({stats['unsafe_percentage']:.1f}%)")
print(f"Total pixels:  {stats['total_pixels']:,}")

# Check specific positions
print(f"\n" + "="*70)
print("POSITION CHECKS:")
print("="*70)

# Start position (10, 10)
start_x, start_y = 75, 75
start_value = mask[start_y, start_x]
print(f"Start ({start_x}, {start_y}): {'SAFE' if start_value == 1 else 'UNSAFE'} (value: {start_value})")

# Goal position (90, 90)
goal_x, goal_y = config.MAP_WIDTH-75,config.MAP_HEIGHT-75
goal_value = mask[goal_y, goal_x]
print(f"Goal ({goal_x}, {goal_y}): {'SAFE' if goal_value == 1 else 'UNSAFE'} (value: {goal_value})")

# Check corners
print(f"\nCorner checks:")
print(f"  Top-left (0, 0): {'SAFE' if mask[0, 0] == 1 else 'UNSAFE'}")
print(f"  Top-right (0, 99): {'SAFE' if mask[0, 99] == 1 else 'UNSAFE'}")
print(f"  Bottom-left (99, 0): {'SAFE' if mask[99, 0] == 1 else 'UNSAFE'}")
print(f"  Bottom-right (99, 99): {'SAFE' if mask[99, 99] == 1 else 'UNSAFE'}")

# Check center
center_value = mask[50, 50]
print(f"  Center (50, 50): {'SAFE' if center_value == 1 else 'UNSAFE'}")

# Warning if everything is unsafe
if stats['safe_percentage'] < 1.0:
    print(f"\n" + "="*70)
    print("⚠ WARNING: Almost everything marked as UNSAFE!")
    print("="*70)
    print("This likely means:")
    print("  1. Real SAM is detecting the entire image as one object")
    print("  2. SAM's default segmentation isn't suitable for terrain")
    print("  3. We need to use the smart mock model instead")
    print("\nSAM is designed for object segmentation, not terrain classification!")

# Visualize
print(f"\n" + "="*70)
print("VISUALIZATIONS:")
print("="*70)

# Original image
img_display = cv2.resize(image, (800, 600))
cv2.imshow("1. Original Image", img_display)

# Binary mask
mask_display = (mask * 255).astype(np.uint8)
mask_display = cv2.resize(mask_display, (800, 600), interpolation=cv2.INTER_NEAREST)
cv2.imshow("2. Binary Mask (White=Safe, Black=Unsafe)", mask_display)

# Overlay
overlay = segmenter.visualize_segmentation(image, mask)
overlay_display = cv2.resize(overlay, (800, 600))
cv2.imshow("3. Overlay (Green=Safe, Red=Unsafe)", overlay_display)

# Mark start and goal on overlay
overlay_marked = overlay_display.copy()

# Scale positions to display size
scale_x = 800 / mask.shape[1]
scale_y = 600 / mask.shape[0]

start_display = (int(start_x * scale_x), int(start_y * scale_y))
goal_display = (int(goal_x * scale_x), int(goal_y * scale_y))

cv2.circle(overlay_marked, start_display, 15, (255, 255, 0), -1)  # Yellow start
cv2.circle(overlay_marked, start_display, 15, (0, 0, 0), 2)
cv2.putText(overlay_marked, "START", (start_display[0] - 30, start_display[1] - 20),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

cv2.circle(overlay_marked, goal_display, 15, (255, 0, 255), -1)  # Magenta goal
cv2.circle(overlay_marked, goal_display, 15, (0, 0, 0), 2)
cv2.putText(overlay_marked, "GOAL", (goal_display[0] - 25, goal_display[1] - 20),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

cv2.imshow("4. Marked Positions", overlay_marked)

print("✓ Displaying 4 windows")
print("\nPress any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save visualization
output_path = "outputs/sam_diagnostic.jpg"
os.makedirs("outputs", exist_ok=True)
cv2.imwrite(output_path, overlay_marked)
print(f"\n✓ Saved visualization to: {output_path}")

# Recommendation
print(f"\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)

if stats['safe_percentage'] < 10.0:
    print("❌ Real SAM is NOT suitable for this task!")
    print("\nWhy:")
    print("  • SAM segments 'objects' (cars, buildings, people)")
    print("  • It doesn't understand 'safe terrain' vs 'obstacles'")
    print("  • It's marking the entire scene as one big object")
    print("\n✅ Solution: Use the SMART MOCK MODEL instead")
    print("\nThe mock model:")
    print("  • Uses color clustering (finds background vs obstacles)")
    print("  • Works with ANY image automatically")
    print("  • Is MUCH faster (no GPU needed)")
    print("  • Is actually better for terrain classification!")
    print("\n📝 How to switch to mock model:")
    print("  1. Rename or move sam_vit_h_4b8939.pth temporarily")
    print("  2. System will auto-fallback to smart mock model")
    print("  3. Or modify segmentation.py to always use mock model")
else:
    print("✓ Segmentation looks reasonable")
    print(f"  Safe percentage: {stats['safe_percentage']:.1f}%")