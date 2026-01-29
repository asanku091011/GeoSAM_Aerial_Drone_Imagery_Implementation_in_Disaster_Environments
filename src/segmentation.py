"""
Segmentation module using SAM (Segment Anything Model) to classify safe and unsafe regions.
Processes drone images and outputs binary masks for navigation.

UPDATED: Now uses real SAM model instead of mock
"""

import cv2
import numpy as np
import time
import torch
from PIL import Image
import config

class GeoSAMSegmenter:
    """
    Applies SAM segmentation to classify terrain as safe or unsafe.
    
    SAM (Segment Anything Model) is a foundation model for semantic segmentation.
    This class wraps the model and provides easy-to-use methods for
    processing drone images in real-time.
    
    UPDATED: Uses real SAM model from Meta AI
    """
    
    def __init__(self):
        """Initialize the SAM model and preprocessing pipeline."""
        self.model = None
        self.predictor = None
        self.device = self._get_device()
        self.is_loaded = False
        
        # Performance tracking
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        
        print(f"Initializing SAM Segmenter on {self.device}...")
    
    def _get_device(self):
        """
        Determine the best device for running the model.
        Raspberry Pi 5 typically uses CPU.
        
        Returns:
            torch.device: The device to use (cuda, mps, or cpu)
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # Apple Silicon
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self):
        """
        Load the SAM model weights.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            print("Loading SAM model...")
            print(f"  Model path: {config.GEOSAM_CHECKPOINT}")
            
            # Check if model file exists
            import os
            if not os.path.exists(config.GEOSAM_CHECKPOINT):
                print(f"✗ Model file not found at: {config.GEOSAM_CHECKPOINT}")
                print(f"  Please download sam_vit_h_4b8939.pth to the models folder")
                print(f"  Falling back to mock model for testing...")
                self.model = self._create_mock_model()
                self.is_loaded = True
                return True
            
            # Load SAM model
            try:
                from segment_anything import sam_model_registry, SamPredictor
                
                print(f"  Loading SAM ViT-H model...")
                sam = sam_model_registry["vit_h"](checkpoint=config.GEOSAM_CHECKPOINT)
                sam.to(device=self.device)
                
                self.predictor = SamPredictor(sam)
                self.model = sam
                self.is_loaded = True
                
                print("✓ Real SAM model loaded successfully!")
                return True
                
            except ImportError:
                print("✗ segment-anything library not installed!")
                print("  Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
                print("  Falling back to mock model for testing...")
                self.model = self._create_mock_model()
                self.is_loaded = True
                return True
            
        except Exception as e:
            print(f"✗ Failed to load model: {str(e)}")
            print(f"  Falling back to mock model for testing...")
            self.model = self._create_mock_model()
            self.is_loaded = True
            return True
    
    def _create_mock_model(self):
        """
        Create a mock model for testing when real SAM is not available.
        This uses traditional CV techniques to simulate segmentation.
        
        Returns:
            callable: Mock model function
        """
        print("⚠ Using mock segmentation model (replace with real SAM)")
        
        def mock_segment(image):
            """
            Improved color-based segmentation - STRICT version.
            Dark areas and low-saturation areas are UNSAFE.
            Only bright, saturated green is SAFE.
            """
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Create safe mask (start with all unsafe)
            safe_mask = np.zeros(gray.shape, dtype=np.uint8)
            
            # Method 1: Detect bright green (safe areas)
            # Must be: Green hue, high saturation, bright value
            lower_green = np.array([35, 60, 120])  # Stricter green range
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Method 2: Brightness check - must be bright
            bright_mask = gray > 100
            
            # Combine: Must be BOTH green AND bright
            safe_mask = cv2.bitwise_and(green_mask, bright_mask.astype(np.uint8) * 255)
            
            # Method 3: Remove very dark areas (obstacles are dark)
            very_dark = gray < 50
            safe_mask[very_dark] = 0
            
            # Clean up with morphology
            kernel = np.ones((3, 3), np.uint8)
            safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_OPEN, kernel)
            safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_CLOSE, kernel)
            
            return safe_mask
        
        return mock_segment
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input.
        
        Args:
            image (numpy.ndarray): Input BGR image from drone
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Resize to model input size
        image = cv2.resize(image, config.SEGMENTATION_INPUT_SIZE)
        
        # Optional: Apply histogram equalization to improve contrast
        if image.mean() < 100 or image.mean() > 200:  # Poor lighting
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            image = cv2.merge([l, a, b])
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        
        return image
    
    def segment(self, image):
        """
        Perform semantic segmentation on an input image.
        
        Args:
            image (numpy.ndarray): Input BGR image from drone
            
        Returns:
            numpy.ndarray: Binary mask where 1=safe, 0=unsafe
        """
        if not self.is_loaded:
            print("⚠ Model not loaded, loading now...")
            if not self.load_model():
                return None
        
        start_time = time.time()
        
        try:
            # Preprocess image
            preprocessed = self.preprocess_image(image)
            
            # Check if we're using real SAM or mock
            if self.predictor is not None:
                # Use REAL SAM model
                mask = self._segment_with_sam(preprocessed)
            else:
                # Use mock model
                mask = self.model(preprocessed)
            
            # Convert to binary: 255 -> 1 (safe), 0 -> 0 (unsafe)
            binary_mask = (mask > 127).astype(np.uint8)
            
            # Resize back to original size if needed
            if binary_mask.shape != (config.MAP_HEIGHT, config.MAP_WIDTH):
                binary_mask = cv2.resize(binary_mask, (config.MAP_WIDTH, config.MAP_HEIGHT),
                                        interpolation=cv2.INTER_NEAREST)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_frames_processed += 1
            self.total_processing_time += processing_time
            
            return binary_mask
            
        except Exception as e:
            print(f"✗ Segmentation error: {str(e)}")
            return None
    
    def _segment_with_sam(self, image):
        """
        Segment using the real SAM model.
        
        Args:
            image (numpy.ndarray): Preprocessed BGR image
            
        Returns:
            numpy.ndarray: Segmentation mask
        """
        # Convert BGR to RGB for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for SAM predictor
        self.predictor.set_image(image_rgb)
        
        # Generate automatic masks
        # For disaster navigation, we want to identify safe vs unsafe regions
        # We'll use SAM in automatic mode to segment everything, then classify
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Generate masks using SAM's automatic mask generation
        # Simple approach: Use center point and let SAM segment
        input_point = np.array([[w//2, h//2]])  # Center of image
        input_label = np.array([1])  # Foreground point
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # Use the mask with highest score
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        
        # Convert boolean mask to uint8
        mask = (mask * 255).astype(np.uint8)
        
        # Post-process: Apply color-based filtering to classify safe/unsafe
        # Green areas = safe, dark areas = unsafe
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect green (safe terrain)
        lower_green = np.array([35, 40, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combine SAM mask with color information
        # Areas that are both segmented AND green = safe
        safe_mask = cv2.bitwise_and(mask, green_mask)
        
        return safe_mask
    
    def segment_with_confidence(self, image):
        """
        Perform segmentation and return confidence scores.
        
        Args:
            image (numpy.ndarray): Input BGR image
            
        Returns:
            tuple: (binary_mask, confidence_map) where confidence is 0-1
        """
        # Get basic segmentation
        mask = self.segment(image)
        
        if mask is None:
            return None, None
        
        # For mock model, create simple confidence based on edge distance
        confidence = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        confidence = confidence / (confidence.max() + 1e-6)  # Normalize to 0-1
        
        # Apply threshold
        low_confidence = confidence < config.SEGMENTATION_CONFIDENCE_THRESHOLD
        mask[low_confidence] = 0  # Mark low-confidence areas as unsafe
        
        return mask, confidence
    
    def visualize_segmentation(self, image, mask):
        """
        Create a visualization overlay of segmentation on original image.
        
        Args:
            image (numpy.ndarray): Original image
            mask (numpy.ndarray): Segmentation mask
            
        Returns:
            numpy.ndarray: Visualization image with overlay
        """
        # Resize image to match mask if needed
        if image.shape[:2] != mask.shape:
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
        
        # Create colored overlay
        overlay = image.copy()
        
        # Safe areas in green
        safe_areas = mask == 1
        overlay[safe_areas] = cv2.addWeighted(
            image[safe_areas], 0.6,
            np.full_like(image[safe_areas], (0, 255, 0)), 0.4,
            0
        )
        
        # Unsafe areas in red
        unsafe_areas = mask == 0
        overlay[unsafe_areas] = cv2.addWeighted(
            image[unsafe_areas], 0.6,
            np.full_like(image[unsafe_areas], (0, 0, 255)), 0.4,
            0
        )
        
        return overlay
    
    def get_statistics(self, mask):
        """
        Calculate statistics about the segmented terrain.
        
        Args:
            mask (numpy.ndarray): Segmentation mask
            
        Returns:
            dict: Statistics including safe/unsafe percentages
        """
        total_pixels = mask.size
        safe_pixels = np.sum(mask == 1)
        unsafe_pixels = total_pixels - safe_pixels
        
        return {
            'total_pixels': total_pixels,
            'safe_pixels': safe_pixels,
            'unsafe_pixels': unsafe_pixels,
            'safe_percentage': (safe_pixels / total_pixels) * 100,
            'unsafe_percentage': (unsafe_pixels / total_pixels) * 100,
            'navigability_score': safe_pixels / total_pixels  # 0-1 score
        }
    
    def get_average_processing_time(self):
        """
        Get average time per frame for segmentation.
        
        Returns:
            float: Average processing time in seconds
        """
        if self.total_frames_processed == 0:
            return 0.0
        return self.total_processing_time / self.total_frames_processed

# Example usage and testing
if __name__ == "__main__":
    print("Testing SAM Segmentation Module")
    print("=" * 50)
    
    # Create segmenter
    segmenter = GeoSAMSegmenter()
    
    # Load model
    if segmenter.load_model():
        # Create test image
        test_image = np.random.randint(0, 255, 
                                       (config.DRONE_FRAME_HEIGHT, 
                                        config.DRONE_FRAME_WIDTH, 3),
                                       dtype=np.uint8)
        
        # Add some green "safe" areas
        cv2.rectangle(test_image, (100, 100), (300, 300), (50, 200, 50), -1)
        cv2.circle(test_image, (500, 400), 80, (60, 220, 60), -1)
        
        print("\nProcessing test image...")
        
        # Segment
        mask = segmenter.segment(test_image)
        
        if mask is not None:
            # Get statistics
            stats = segmenter.get_statistics(mask)
            print(f"\n✓ Segmentation successful!")
            print(f"  Safe terrain: {stats['safe_percentage']:.1f}%")
            print(f"  Unsafe terrain: {stats['unsafe_percentage']:.1f}%")
            print(f"  Processing time: {segmenter.get_average_processing_time()*1000:.1f}ms")
            
            # Visualize
            viz = segmenter.visualize_segmentation(test_image, mask)
            
            # Display results
            cv2.imshow("Original", test_image)
            cv2.imshow("Segmentation Mask", mask * 255)
            cv2.imshow("Overlay", viz)
            
            print("\nPress any key to close windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("✗ Segmentation failed")
    
    else:
        print("✗ Failed to load model")
