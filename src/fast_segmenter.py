"""
Fast Terrain Segmenter - Optimized for dynamic scene generator
Uses traditional computer vision for quick terrain classification.
"""

import cv2
import numpy as np
import time


class FastSegmenter:
    """
    Fast segmenter using color-based computer vision.
    Optimized for dynamic disaster scenes.
    """
    
    def __init__(self):
        self.is_loaded = False
        print("Initializing Fast Segmenter (traditional CV)...")
    
    def load_model(self):
        """No model to load - just mark as ready."""
        print("✓ Fast segmenter ready (no model loading needed)")
        self.is_loaded = True
        return True
    
    def segment(self, image):
        """
        Segment terrain using color-based analysis.
        Optimized for dynamic disaster scenes.
        
        Args:
            image (numpy.ndarray): BGR image
            
        Returns:
            numpy.ndarray: Binary mask (1=safe, 0=obstacle)
        """
        if not self.is_loaded:
            print("⚠ Segmenter not loaded")
            return None
        
        start_time = time.time()
        
        try:
            # Work at full resolution for accuracy
            img = image.copy()
            h, w = img.shape[:2]
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            
            # Start with all safe
            safe_mask = np.ones(gray.shape, dtype=np.uint8)
            
            # === OBSTACLE DETECTION FOR DYNAMIC SCENES ===
            
            # 1. Very dark obstacles (debris, dark concrete)
            # This catches obstacles in 10-30 range
            very_dark_mask = gray < 40
            safe_mask[very_dark_mask] = 0
            
            # 2. Dark-medium obstacles (gray concrete)
            # This catches obstacles in 60-90 range
            # But exclude if it's part of natural ground variation
            medium_dark_mask = (gray >= 40) & (gray < 100)
            # Only mark as obstacle if it's relatively uniform (not textured ground)
            # Check local standard deviation
            gray_float = gray.astype(float)
            kernel_size = 15
            local_mean = cv2.blur(gray_float, (kernel_size, kernel_size))
            local_sq_mean = cv2.blur(gray_float**2, (kernel_size, kernel_size))
            local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
            
            # Low local variation + medium dark = concrete obstacle
            uniform_dark = medium_dark_mask & (local_std < 15)
            safe_mask[uniform_dark] = 0
            
            # 3. Low saturation + dark (gray obstacles)
            # Catches desaturated obstacles
            gray_obstacle_mask = (s_channel < 30) & (gray < 110) & (gray > 30)
            safe_mask[gray_obstacle_mask] = 0
            
            # 4. Sky detection (if present) - very bright + top portion
            # But be careful not to catch white start/goal circles
            sky_mask = (v_channel > 200) & (s_channel < 40)
            
            # Only mark as obstacle if in top 30% of image
            y_coords = np.arange(h)[:, np.newaxis]
            top_region = y_coords < (h * 0.3)
            
            sky_final = sky_mask & top_region
            safe_mask[sky_final] = 0
            
            # 5. Blue water detection (if present)
            water_mask = (h_channel > 90) & (h_channel < 130) & (s_channel > 40)
            safe_mask[water_mask] = 0
            
            # 6. IMPORTANT: Restore white circles (start/goal positions)
            # These should always be safe
            white_mask = (gray > 240) & (s_channel < 20)
            
            # Find circular white regions (start/goal markers)
            # These are bright, low saturation, and roughly circular
            kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            white_dilated = cv2.dilate(white_mask.astype(np.uint8), kernel_circle)
            
            # Mark white circles as safe
            safe_mask[white_dilated > 0] = 1
            
            # === POST-PROCESSING ===
            
            # Remove small noise
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_OPEN, kernel_small)
            
            # Fill small holes
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_CLOSE, kernel_medium)
            
            # Final smoothing
            safe_mask = cv2.GaussianBlur(safe_mask.astype(float), (5, 5), 0)
            safe_mask = (safe_mask > 0.5).astype(np.uint8)
            
            elapsed = time.time() - start_time
            print(f"  Fast segmentation took {elapsed:.2f}s")
            
            # Get stats
            safe_pct = (np.sum(safe_mask == 1) / safe_mask.size) * 100
            print(f"  Result: {safe_pct:.1f}% safe terrain")
            
            return safe_mask
            
        except Exception as e:
            print(f"✗ Segmentation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_segmentation(self, image, mask):
        """Create visualization overlay."""
        if image.shape[:2] != mask.shape:
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
        
        overlay = image.copy()
        
        # Red for obstacles (0)
        red = np.array([0, 0, 255])  # BGR format
        obstacle_mask = mask == 0
        if np.any(obstacle_mask):
            overlay[obstacle_mask] = (
                0.6 * overlay[obstacle_mask] + 0.4 * red
            ).astype(np.uint8)
        
        # Green for safe terrain (1)
        green = np.array([0, 255, 0])  # BGR format
        safe_mask = mask == 1
        if np.any(safe_mask):
            overlay[safe_mask] = (
                0.6 * overlay[safe_mask] + 0.4 * green
            ).astype(np.uint8)
        
        return overlay
    
    def get_statistics(self, mask):
        """Get mask statistics."""
        total_pixels = mask.size
        safe_pixels = np.sum(mask == 1)
        unsafe_pixels = total_pixels - safe_pixels
        
        return {
            'total_pixels': total_pixels,
            'safe_pixels': safe_pixels,
            'unsafe_pixels': unsafe_pixels,
            'safe_percentage': (safe_pixels / total_pixels) * 100,
            'unsafe_percentage': (unsafe_pixels / total_pixels) * 100,
            'navigability_score': safe_pixels / total_pixels
        }


# Test
# Test
if __name__ == "__main__":
    print("Testing Fast Segmenter")
    print("="*70)
    
    import os
    try:
        import config
        TEST_IMAGE = config.TEST_IMAGE_PATH
    except:
        TEST_IMAGE = "data/test_images/testearthquake.jpg"
    
    print(f"\nTest image: {TEST_IMAGE}")
    print("(Configure in config.py: TEST_IMAGE_PATH)")
    
    # Check if image exists
    if not os.path.exists(TEST_IMAGE):
        print(f"\n⚠ Test image not found: {TEST_IMAGE}")
        print("\nOptions:")
        print("1. Set config.TEST_IMAGE_PATH to a valid image")
        print("2. Use 'data/test_images/current_scene.jpg' for dynamic scenes")
        print("3. Use 'data/test_images/ladi_XXXXX.jpg' for LADI dataset")
        exit(1)
    
    segmenter = FastSegmenter()
    
    if segmenter.load_model():
        # Load test image
        image = cv2.imread(TEST_IMAGE)
        print(f"Image shape: {image.shape}")
        
        # Segment
        mask = segmenter.segment(image)
        
        if mask is not None:
            # Get stats
            stats = segmenter.get_statistics(mask)
            print(f"\n{'='*70}")
            print("RESULTS:")
            print(f"{'='*70}")
            print(f"Safe terrain: {stats['safe_percentage']:.1f}%")
            print(f"Unsafe terrain: {stats['unsafe_percentage']:.1f}%")
            
            # Create binary visualization
            binary_viz = (mask * 255).astype(np.uint8)
            
            # Visualize
            overlay = segmenter.visualize_segmentation(image, mask)
            
            cv2.imshow("Original", cv2.resize(image, (800, 600)))
            cv2.imshow("Binary Mask (Black=Obstacle, White=Safe)", cv2.resize(binary_viz, (800, 600)))
            cv2.imshow("Segmentation Overlay", cv2.resize(overlay, (800, 600)))
            
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("✗ Failed to load segmenter")