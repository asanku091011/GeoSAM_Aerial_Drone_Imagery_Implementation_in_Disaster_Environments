"""
Segmentation module using Geo-SAM to classify safe and unsafe regions.
Processes drone images and outputs binary masks for navigation.
"""

import cv2
import numpy as np
import time
import torch
from PIL import Image
import config

class GeoSAMSegmenter:
    """
    Applies Geo-SAM segmentation to classify terrain as safe or unsafe.
    
    Geo-SAM is a geospatial foundation model for semantic segmentation.
    This class wraps the model and provides easy-to-use methods for
    processing drone images in real-time.
    """
    
    def __init__(self):
        """Initialize the Geo-SAM model and preprocessing pipeline."""
        self.model = None
        self.device = self._get_device()
        self.is_loaded = False
        
        # Performance tracking
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        
        print(f"Initializing Geo-SAM Segmenter on {self.device}...")
    
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
        Load the Geo-SAM model weights.
        
        NOTE: This is where you would load the actual Geo-SAM model.
        You need to download the model weights first.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            print("Loading Geo-SAM model...")
            
            # PLACEHOLDER: Load your Geo-SAM model here
            # Example (you'll need to adapt this to actual Geo-SAM):
            # from geosam import GeoSAM
            # self.model = GeoSAM.from_pretrained(config.GEOSAM_MODEL_PATH)
            # self.model.to(self.device)
            # self.model.eval()
            
            # For now, we'll create a mock model for testing
            print("⚠ Using mock segmentation model (replace with real Geo-SAM)")
            self.model = self._create_mock_model()
            self.is_loaded = True
            
            print("✓ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load model: {str(e)}")
            return False
    
    def _create_mock_model(self):
        """
        Create a mock model for testing when real Geo-SAM is not available.
        This uses traditional CV techniques to simulate segmentation.
        
        Returns:
            callable: Mock model function
        """
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
            
            # Run segmentation
            # For real Geo-SAM, this would be:
            # with torch.no_grad():
            #     input_tensor = self._to_tensor(preprocessed)
            #     output = self.model(input_tensor)
            #     mask = self._postprocess_output(output)
            
            # Using mock model:
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
    print("Testing Geo-SAM Segmentation Module")
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