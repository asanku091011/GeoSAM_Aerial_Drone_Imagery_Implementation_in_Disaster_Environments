"""
Debug segmentation to see exactly what's happening
"""
import cv2
import numpy as np
import os
from segmentation import GeoSAMSegmenter
from map_builder import MapBuilder

print("="*70)
print("SEGMENTATION DEBUG")
print("="*70)

# Load the test image (Windows compatible path)
image_path = os.path.join("..", "src", "data", "test_images", "ladi_01027_segmented.jpg")
#image_path = os.path.join("..", "src", "data", "test_images", "current_scene.jpg")
#image_path = os.path.join("..", "src", "data", "test_images", "tello_current.jpg")
print(f"\nLoading image: {image_path}")
print(f"Full path: {os.path.abspath(image_path)}")

image = cv2.imread(image_path)

if image is None:
    print("✗ Image not found! Creating it now...")
    from image_input import ImageInput
    img_input = ImageInput()
    img_input.connect()  # This will create default_scene.jpg
    
    # Try loading again
    image = cv2.imread(image_path)
    
    if image is None:
        print("✗ Still can't load image!")
        print(f"Looking for: {os.path.abspath(image_path)}")
        print("\nTry running this first:")
        print("  python image_input.py")
        exit()

print(f"✓ Image loaded: {image.shape}")
print(f"  Image stats: min={image.min()}, max={image.max()}, mean={image.mean():.1f}")

# Analyze the image colors
print("\nAnalyzing image colors...")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(f"  Gray values: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
print(f"  HSV H: min={hsv[:,:,0].min()}, max={hsv[:,:,0].max()}")
print(f"  HSV S: min={hsv[:,:,1].min()}, max={hsv[:,:,1].max()}")
print(f"  HSV V: min={hsv[:,:,2].min()}, max={hsv[:,:,2].max()}")

# Count dark pixels (should be obstacles)
dark_pixels = np.sum(gray < 50)
total_pixels = gray.size
print(f"\nDark pixels (< 50): {dark_pixels} ({dark_pixels/total_pixels*100:.1f}%)")

# Create segmenter
print("\nCreating segmenter...")
segmenter = GeoSAMSegmenter()
segmenter.load_model()

# Segment the image
print("\nSegmenting image...")
mask = segmenter.segment(image)

if mask is None:
    print("✗ Segmentation failed!")
    exit()

print(f"✓ Segmentation complete")
print(f"  Mask shape: {mask.shape}")
print(f"  Unique values: {np.unique(mask)}")

# Get statistics
stats = segmenter.get_statistics(mask)
print(f"\nSegmentation results:")
print(f"  Safe pixels: {stats['safe_pixels']} ({stats['safe_percentage']:.1f}%)")
print(f"  Unsafe pixels: {stats['unsafe_pixels']} ({stats['unsafe_percentage']:.1f}%)")

# Create visualization
overlay = segmenter.visualize_segmentation(image, mask)

# Also show the raw mask
mask_vis = mask * 255
mask_vis_color = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

# Build the map
print("\nBuilding navigation map...")
map_builder = MapBuilder()
map_builder.update_from_segmentation(mask)

map_stats = map_builder.get_statistics()
print(f"  Free cells: {map_stats['free_cells']} ({map_stats['free_percentage']:.1f}%)")
print(f"  Obstacle cells: {map_stats['obstacle_cells']} ({map_stats['obstacle_percentage']:.1f}%)")

# Show all the visualizations
print("\n" + "="*70)
print("DISPLAYING VISUALIZATIONS")
print("="*70)
print("Windows:")
print("  1. Original Image")
print("  2. Segmentation Overlay (green=safe, red=unsafe)")
print("  3. Binary Mask (white=safe, black=unsafe)")
print("  4. Navigation Grid")
print("\nPress any key to close...")
print("="*70)

cv2.imshow("1. Original Image", image)
cv2.imshow("2. Segmentation Overlay", overlay)
cv2.imshow("3. Binary Mask", mask_vis_color)

# Show the grid
grid_vis = map_builder.visualize()
cv2.imshow("4. Navigation Grid", grid_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n✓ Debug complete!")

# Print recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)

if stats['safe_percentage'] > 85:
    print("⚠ WARNING: Too much safe terrain detected!")
    print("  The segmentation is marking obstacles as safe.")
    print("  Solutions:")
    print("  1. Use real SAM model (if you have the weights)")
    print("  2. Adjust segmentation thresholds in segmentation.py")
    print("  3. Use different test images with more contrast")
elif stats['safe_percentage'] < 50:
    print("⚠ WARNING: Too little safe terrain detected!")
    print("  The segmentation is too strict.")
    print("  Solution: Relax thresholds in segmentation.py")
else:
    print("✓ Segmentation looks reasonable!")
    print("  Safe terrain: {:.1f}%".format(stats['safe_percentage']))