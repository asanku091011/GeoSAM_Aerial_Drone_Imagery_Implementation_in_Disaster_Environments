"""
Image input module for using still images instead of drone video.
Perfect for testing with photos of disaster scenes.
"""

import cv2
import numpy as np
import os
import glob
from threading import Thread, Lock
import time
import config

class ImageInput:
    """
    Loads and provides still images as if they were from a drone.
    Can use a single image or loop through multiple images.
    
    Usage:
    1. Put images in data/test_images/
    2. Set DRONE_ENABLED = False and TEST_MODE = False in config
    3. Run system - it will use your images!
    """
    
    def __init__(self, image_folder="data/test_images"):
        """
        Initialize image input.
        
        Args:
            image_folder (str): Folder containing test images
        """
        self.image_folder = image_folder
        self.images = []
        self.current_image = None
        self.current_index = 0
        self.frame_lock = Lock()
        self.is_streaming = False
        self.capture_thread = None
        
        # Performance metrics
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        
        print(f"Initializing image input from: {image_folder}")
    
    def connect(self):
        """
        Load images from folder.
        
        Returns:
            bool: True if images loaded successfully
        """
        try:
            # Create folder if it doesn't exist
            os.makedirs(self.image_folder, exist_ok=True)
            
            # Find all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for ext in image_extensions:
                pattern = os.path.join(self.image_folder, ext)
                self.images.extend(glob.glob(pattern))
            
            if not self.images:
                print(f"⚠ No images found in {self.image_folder}")
                print(f"  Generating a synthetic test image...")
                self._create_default_image()
                return True
            
            # Sort images for consistent ordering
            self.images.sort()
            
            print(f"✓ Found {len(self.images)} image(s):")
            for i, img in enumerate(self.images[:5]):  # Show first 5
                print(f"  {i+1}. {os.path.basename(img)}")
            if len(self.images) > 5:
                print(f"  ... and {len(self.images) - 5} more")
            
            # Load first image
            self.current_image = cv2.imread(self.images[0])
            
            if self.current_image is None:
                print(f"✗ Failed to load {self.images[0]}")
                return False
            
            print(f"✓ Images loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load images: {str(e)}")
            return False
    
    def _create_default_image(self):
        """Create a default test image if none exist."""
        print("Creating default disaster scene image...")
        
        width, height = config.DRONE_FRAME_WIDTH, config.DRONE_FRAME_HEIGHT
        
        # Create image with VERY bright green background (clearly safe)
        image = np.ones((height, width, 3), dtype=np.uint8)
        image[:, :] = (80, 255, 80)  # BRIGHT green = very obviously safe
        
        # Calculate start and goal positions (will just be normal green, no markers)
        start_x, start_y = int(width * 0.10), int(height * 0.10)
        goal_x, goal_y = int(width * 0.90), int(height * 0.90)
        
        # Create obstacles that are VERY DARK and clearly different from green
        # Position them to force interesting paths
        obstacles = [
            # Vertical wall on left side (forces path to go around)
            (int(width * 0.30), int(height * 0.20), 60, 400),
            
            # Horizontal wall in middle
            (int(width * 0.45), int(height * 0.50), 300, 60),
            
            # Additional scattered obstacles
            (int(width * 0.65), int(height * 0.25), 90, 90),
            (int(width * 0.20), int(height * 0.65), 80, 80),
            (int(width * 0.75), int(height * 0.70), 85, 85),
        ]
        
        for x, y, w, h in obstacles:
            # Very dark obstacles (almost black)
            cv2.rectangle(image, (x, y), (x + w, y + h), (15, 15, 15), -1)
            # Red border to make them VERY obvious
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        # Add subtle text labels for start/goal (small, won't affect segmentation)
        cv2.putText(image, "S", (start_x - 10, start_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
        cv2.putText(image, "G", (goal_x - 10, goal_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
        
        # Add instruction text at top
        cv2.putText(image, "Disaster Scene - Black = Obstacles", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Save it
        filepath = os.path.join(self.image_folder, "default_scene.jpg")
        cv2.imwrite(filepath, image)
        
        self.images = [filepath]
        self.current_image = image
        
        print(f"✓ Created default image at: {filepath}")
        print(f"  Image has bright green safe areas and dark black obstacles")
        print(f"  Start position (10%, 10%) and Goal (90%, 90%) are in green areas")
    
    def start_stream(self):
        """
        Start providing frames (just returns same image or cycles through images).
        """
        if self.is_streaming:
            print("Stream already running")
            return
        
        self.is_streaming = True
        self.capture_thread = Thread(target=self._update_loop, daemon=True)
        self.capture_thread.start()
        print("✓ Image provider started")
    
    def _update_loop(self):
        """
        Background thread that updates current image.
        Can cycle through multiple images or just hold one.
        """
        while self.is_streaming:
            try:
                # Update FPS counter
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_frame_time = current_time
                
                # Optionally cycle through images (if multiple)
                # Uncomment this to auto-cycle every 5 seconds:
                # if len(self.images) > 1 and elapsed >= 5.0:
                #     self.current_index = (self.current_index + 1) % len(self.images)
                #     self.current_image = cv2.imread(self.images[self.current_index])
                
                time.sleep(1.0 / config.DRONE_FPS)
                
            except Exception as e:
                print(f"⚠ Update loop error: {str(e)}")
                time.sleep(0.1)
    
    def get_frame(self):
        """
        Get current image frame.
        
        Returns:
            numpy.ndarray: BGR image
        """
        with self.frame_lock:
            if self.current_image is not None:
                return self.current_image.copy()
            else:
                return None
    
    def get_fps(self):
        """Get simulated FPS (for compatibility)."""
        return self.fps
    
    def switch_image(self, index):
        """
        Switch to a different image by index.
        
        Args:
            index (int): Image index to switch to
        """
        if 0 <= index < len(self.images):
            self.current_index = index
            self.current_image = cv2.imread(self.images[index])
            print(f"Switched to image: {os.path.basename(self.images[index])}")
    
    def next_image(self):
        """Switch to next image in folder."""
        if len(self.images) > 1:
            self.current_index = (self.current_index + 1) % len(self.images)
            self.switch_image(self.current_index)
    
    def previous_image(self):
        """Switch to previous image in folder."""
        if len(self.images) > 1:
            self.current_index = (self.current_index - 1) % len(self.images)
            self.switch_image(self.current_index)
    
    def stop_stream(self):
        """Stop the update loop."""
        print("Stopping image provider...")
        self.is_streaming = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        print("✓ Stopped")
    
    def disconnect(self):
        """Cleanup."""
        self.stop_stream()
        print("✓ Image input disconnected")
    
    # Compatibility methods (so it works like DroneInput)
    def takeoff(self):
        """Dummy method for compatibility."""
        pass
    
    def land(self):
        """Dummy method for compatibility."""
        pass
    
    def emergency_stop(self):
        """Dummy method for compatibility."""
        pass


# Testing
if __name__ == "__main__":
    print("Testing Image Input Module")
    print("=" * 50)
    
    # Create image input
    img_input = ImageInput()
    
    # Connect (load images)
    if img_input.connect():
        # Start providing frames
        img_input.start_stream()
        
        print("\nDisplaying image for 10 seconds...")
        print("Press 'Q' to quit early")
        print("Press 'N' for next image, 'P' for previous")
        
        start_time = time.time()
        
        while time.time() - start_time < 10:
            frame = img_input.get_frame()
            
            if frame is not None:
                # Add info overlay
                cv2.putText(frame, f"Image {img_input.current_index + 1}/{len(img_input.images)}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "N=Next  P=Previous  Q=Quit",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("Test Image", frame)
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    img_input.next_image()
                elif key == ord('p'):
                    img_input.previous_image()
        
        cv2.destroyAllWindows()
        img_input.disconnect()
        
        print("\n✓ Test complete!")
    else:
        print("✗ Failed to load images")