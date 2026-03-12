"""
MobileSAM Segmenter - FIXED VERSION
Works with the actual MobileSAM installation
"""

import cv2
import numpy as np
import time
import torch
import config

class MobileSAMSegmenter:
    """
    MobileSAM implementation for fast terrain segmentation.
    
    MobileSAM is ~10x faster than regular SAM while maintaining
    95% of the accuracy - perfect for real-time navigation demos!
    """
    
    def __init__(self):
        self.model = None
        self.predictor = None
        self.device = self._get_device()
        self.is_loaded = False
        
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        
        print(f"Initializing MobileSAM Segmenter on {self.device}...")
    
    def _get_device(self):
        """Get best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self):
        """Load MobileSAM model."""
        try:
            print("Loading MobileSAM model...")
            print("  Note: MobileSAM is ~10x faster than regular SAM!")
            
            # Check for model checkpoint
            import os
            if not os.path.exists(config.MOBILESAM_CHECKPOINT):
                print(f"✗ MobileSAM checkpoint not found!")
                print(f"  Looking for: {config.MOBILESAM_CHECKPOINT}")
                print("\nDownload from:")
                print("  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt")
                print(f"  Save to: {config.MOBILESAM_CHECKPOINT}")
                print("\nFalling back to fast segmenter...")
                self._fallback_to_fast_segmenter()
                return True
            
            # CORRECT import for MobileSAM
            from mobile_sam import sam_model_registry, SamPredictor
            
            # Load MobileSAM using vit_t
            start_time = time.time()
            sam = sam_model_registry["vit_t"](checkpoint=config.MOBILESAM_CHECKPOINT)
            sam.to(device=self.device)
            
            self.predictor = SamPredictor(sam)
            self.model = sam
            self.is_loaded = True
            
            elapsed = time.time() - start_time
            print(f"✓ MobileSAM loaded in {elapsed:.2f}s!")
            print(f"  Model: ViT-Tiny (lightweight)")
            print(f"  Expected segmentation time: 3-8s per image on CPU")
            print(f"  This is ~10x faster than regular SAM!")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load MobileSAM: {e}")
            import traceback
            traceback.print_exc()
            print("  Falling back to fast segmenter...")
            self._fallback_to_fast_segmenter()
            return True
    
    def _fallback_to_fast_segmenter(self):
        """Fallback to Fast Segmenter if MobileSAM fails."""
        from fast_segmenter import FastSegmenter
        fallback = FastSegmenter()
        fallback.load_model()
        # Replace this object with fast segmenter
        self.__class__ = fallback.__class__
        self.__dict__ = fallback.__dict__
    
    def segment(self, image):
        """
        Segment image using MobileSAM.
        
        Uses grid-based prompting for fast terrain classification.
        """
        if not self.is_loaded:
            print("⚠ Model not loaded, loading now...")
            if not self.load_model():
                return None
        
        start_time = time.time()
        
        try:
            # Resize for faster processing
            h, w = image.shape[:2]
            max_dim = 640
            scale = max_dim / max(h, w)
            
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                image_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                image_small = image.copy()
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
            
            print(f"  Segmenting {image_small.shape[1]}x{image_small.shape[0]} image with MobileSAM...")
            
            # Set image in predictor
            self.predictor.set_image(image_rgb)
            
            # Create grid of sample points for classification
            h_small, w_small = image_small.shape[:2]
            
            # Sample points in a grid (fewer points = faster)
            grid_size = 16  # 16x16 grid
            grid_points = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    x = int((j + 0.5) * w_small / grid_size)
                    y = int((i + 0.5) * h_small / grid_size)
                    grid_points.append([x, y])
            
            grid_points = np.array(grid_points)
            grid_labels = np.ones(len(grid_points), dtype=int)
            
            # Predict with MobileSAM
            masks, scores, _ = self.predictor.predict(
                point_coords=grid_points,
                point_labels=grid_labels,
                multimask_output=False
            )
            
            # Combine masks
            combined_mask = np.zeros((h_small, w_small), dtype=np.uint8)
            for mask in masks:
                combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))
            
            # Classify terrain using color analysis
            terrain_mask = self._classify_terrain(image_small, combined_mask)
            
            # Resize to target size
            final_mask = cv2.resize(terrain_mask, (config.MAP_WIDTH, config.MAP_HEIGHT),
                                   interpolation=cv2.INTER_NEAREST)
            
            # Update metrics
            elapsed = time.time() - start_time
            self.total_frames_processed += 1
            self.total_processing_time += elapsed
            
            print(f"  MobileSAM segmentation took {elapsed:.2f}s")
            
            return final_mask
            
        except Exception as e:
            print(f"✗ Segmentation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _classify_terrain(self, image, mask):
        """
        Classify segmented regions as safe/unsafe terrain.
        
        Uses color analysis to distinguish obstacles from safe ground.
        """
        # Start with all safe
        terrain_mask = np.ones(mask.shape, dtype=np.uint8)
        
        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Dark regions = obstacles
        dark_threshold = 80
        dark_regions = gray < dark_threshold
        terrain_mask[dark_regions] = 0
        
        # Use K-means clustering for better results
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3).astype(np.float32)
        
        # K-means with 3 clusters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
        _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        
        labels_2d = labels.reshape(mask.shape)
        
        # Find darkest cluster (obstacles)
        label_brightness = [gray[labels_2d == i].mean() for i in range(3)]
        darkest_label = np.argmin(label_brightness)
        
        # Mark darkest cluster as unsafe
        terrain_mask[labels_2d == darkest_label] = 0
        
        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        terrain_mask = cv2.morphologyEx(terrain_mask, cv2.MORPH_OPEN, kernel)
        terrain_mask = cv2.morphologyEx(terrain_mask, cv2.MORPH_CLOSE, kernel)
        
        return terrain_mask
    
    def visualize_segmentation(self, image, mask):
        """Create visualization overlay."""
        if image.shape[:2] != mask.shape:
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
        
        overlay = image.copy()
        
        # Safe = green
        safe_areas = mask == 1
        overlay[safe_areas] = cv2.addWeighted(
            image[safe_areas], 0.6,
            np.full_like(image[safe_areas], (0, 255, 0)), 0.4, 0
        )
        
        # Unsafe = red
        unsafe_areas = mask == 0
        overlay[unsafe_areas] = cv2.addWeighted(
            image[unsafe_areas], 0.6,
            np.full_like(image[unsafe_areas], (0, 0, 255)), 0.4, 0
        )
        
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
    
    def get_average_processing_time(self):
        """Get average processing time."""
        if self.total_frames_processed == 0:
            return 0.0
        return self.total_processing_time / self.total_frames_processed


# Test
if __name__ == "__main__":
    print("Testing MobileSAM Segmenter")
    print("="*70)
    
    segmenter = MobileSAMSegmenter()
    
    if segmenter.load_model():
        # Load test image
        import os
        test_path = "data/test_images/DJIDrone.jpg"
        
        if os.path.exists(test_path):
            image = cv2.imread(test_path)
            print(f"\nSegmenting: {test_path}")
            
            mask = segmenter.segment(image)
            
            if mask is not None:
                stats = segmenter.get_statistics(mask)
                print(f"\n{'='*70}")
                print("RESULTS:")
                print(f"{'='*70}")
                print(f"Safe terrain: {stats['safe_percentage']:.1f}%")
                print(f"Processing time: {segmenter.get_average_processing_time():.2f}s")
                
                # Visualize
                overlay = segmenter.visualize_segmentation(image, mask)
                cv2.imshow("MobileSAM Segmentation", cv2.resize(overlay, (800, 600)))
                print("\nPress any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"✗ Test image not found: {test_path}")
    else:
        print("✗ Failed to load MobileSAM")