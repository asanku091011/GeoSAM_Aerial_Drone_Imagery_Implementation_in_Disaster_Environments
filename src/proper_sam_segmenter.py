"""
Accurate SAM terrain segmentation (based on your working code)
Uses SamAutomaticMaskGenerator and inverts the result
"""

import cv2
import numpy as np
import torch
import time
import config
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class AccurateSAMSegmenter:
    """
    Uses SAM's automatic mask generation like your working code.
    All detected segments = obstacles
    Everything else = safe terrain
    """
    
    def __init__(self):
        self.model = None
        self.predictor = None
        self.mask_generator = None
        self.device = self._get_device()
        self.is_loaded = False
        
        print(f"Initializing Accurate SAM Segmenter on {self.device}...")
    
    def _get_device(self):
        """Get best device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self):
        """Load SAM with automatic mask generation."""
        try:
            print("Loading SAM for automatic mask generation...")
            
            import os
            if not os.path.exists(config.GEOSAM_CHECKPOINT):
                print(f"✗ SAM model not found at: {config.GEOSAM_CHECKPOINT}")
                return False
            
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            # Load SAM
            sam = sam_model_registry["vit_h"](checkpoint=config.GEOSAM_CHECKPOINT)
            sam.to(device=self.device)
            
            # Create mask generator (same as your working code)
            self.mask_generator = SamAutomaticMaskGenerator(sam)
            
            self.model = sam
            self.is_loaded = True
            
            print("✓ SAM automatic mask generator loaded!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load SAM: {e}")
            return False
    
    def segment(self, image):
        """
        Segment terrain using SAM automatic mask generation.
        This matches your working code's approach.
        
        Args:
            image (numpy.ndarray): BGR image
            
        Returns:
            numpy.ndarray: Binary mask (1=safe, 0=obstacle)
        """
        if not self.is_loaded:
            print("⚠ Model not loaded")
            return None
        
        start_time = time.time()
        
        try:
            # Convert to RGB (SAM expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            print(f"  Generating masks with SAM on {image_rgb.shape[:2]} image...")
            
            # Generate all masks (just like your working code)
            masks = self.mask_generator.generate(image_rgb)
            
            print(f"  SAM found {len(masks)} segments")
            
            # Create segmentation mask - start with zeros
            segmentation_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            
            # Mark all detected segments as 1
            for mask_data in masks:
                segmentation_mask[mask_data['segmentation']] = 1
            
            # INVERT - this is the key from your working code!
            # All detected objects become 0 (obstacles)
            # Everything else becomes 1 (safe terrain)
            segmentation_mask = 1 - segmentation_mask
            
            # Resize to target size
            final_mask = cv2.resize(
                segmentation_mask, 
                (config.MAP_WIDTH, config.MAP_HEIGHT),
                interpolation=cv2.INTER_NEAREST
            )
            
            elapsed = time.time() - start_time
            print(f"  Segmentation took {elapsed:.2f}s")
            
            # Get stats
            safe_pct = (np.sum(final_mask == 1) / final_mask.size) * 100
            print(f"  Result: {safe_pct:.1f}% safe terrain")
            
            return final_mask
            
        except Exception as e:
            print(f"✗ Segmentation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_segmentation(self, image, mask):
        """Create visualization overlay (matches your working code)."""
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
    print("Testing Accurate SAM Segmenter")
    print("="*70)
    
    segmenter = AccurateSAMSegmenter()
    
    if segmenter.load_model():
        import os
        test_image_path = "data/test_images/testearthquake.jpg"
        
        if not os.path.exists(test_image_path):
            print(f"✗ Test image not found: {test_image_path}")
            exit(1)
        
        image = cv2.imread(test_image_path)
        print(f"\nSegmenting: {test_image_path}")
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
            
            # Create binary visualization (black = obstacle, white = safe)
            binary_viz = (mask * 255).astype(np.uint8)
            
            # Visualize
            overlay = segmenter.visualize_segmentation(image, mask)
            
            # Show all three windows
            cv2.imshow("Original", cv2.resize(image, (800, 600)))
            cv2.imshow("Binary Mask (Black=Obstacle, White=Safe)", cv2.resize(binary_viz, (800, 600)))
            cv2.imshow("Segmentation Overlay", cv2.resize(overlay, (800, 600)))
            
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("✗ Failed to load SAM")