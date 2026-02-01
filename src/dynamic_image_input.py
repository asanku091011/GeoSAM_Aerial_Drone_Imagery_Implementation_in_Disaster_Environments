"""
Modified ImageInput - Reloads image file each time to detect changes
Add this to your image_input.py or use as replacement
"""

import cv2
import numpy as np
import os
import time
from threading import Thread, Lock
import config


class DynamicImageInput:
    """
    Image input that RELOADS the file each time get_frame() is called.
    This allows external processes to update the image file.
    
    Perfect for testing dynamic replanning with changing images!
    """
    
    def __init__(self, image_path="data/test_images/current_scene.jpg"):
        """
        Initialize dynamic image input.
        
        Args:
            image_path (str): Path to image file (will be reloaded each call)
        """
        self.image_path = image_path
        self.current_image = None
        self.frame_lock = Lock()
        self.is_streaming = False
        
        # Performance metrics
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        
        # File change detection
        self.last_modified = 0
        
        print(f"Dynamic Image Input initialized")
        print(f"  Watching: {image_path}")
        print(f"  Mode: Reloads file each frame (detects changes)")
    
    def connect(self):
        """
        Initialize - verify image file exists or create it.
        
        Returns:
            bool: True if successful
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.image_path), exist_ok=True)
            
            # Check if file exists
            if not os.path.exists(self.image_path):
                print(f"⚠ Image not found: {self.image_path}")
                print(f"  Creating default scene...")
                self._create_default_scene()
            
            # Try to load it
            self.current_image = cv2.imread(self.image_path)
            
            if self.current_image is None:
                print(f"✗ Failed to load {self.image_path}")
                return False
            
            self.last_modified = os.path.getmtime(self.image_path)
            
            print(f"✓ Image loaded: {self.image_path}")
            print(f"  Size: {self.current_image.shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect: {str(e)}")
            return False
    
    def _create_default_scene(self):
        """Create a default scene image if none exists."""
        width, height = config.DRONE_FRAME_WIDTH, config.DRONE_FRAME_HEIGHT
        
        # Bright green background
        image = np.ones((height, width, 3), dtype=np.uint8)
        image[:, :] = (80, 255, 80)
        
        # Add some obstacles
        obstacles = [
            (int(width * 0.20), int(height * 0.30), 80, 100),
            (int(width * 0.65), int(height * 0.30), 75, 110),
            (int(width * 0.40), int(height * 0.60), 90, 90),
        ]
        
        for x, y, w, h in obstacles:
            cv2.rectangle(image, (x, y), (x + w, y + h), (15, 15, 15), -1)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Mark start and goal
        start_x = int(width * 0.10)
        start_y = int(height * 0.10)
        goal_x = int(width * 0.90)
        goal_y = int(height * 0.90)
        
        cv2.circle(image, (start_x, start_y), 40, (50, 255, 50), -1)
        cv2.circle(image, (goal_x, goal_y), 40, (50, 255, 50), -1)
        
        cv2.putText(image, "S", (start_x - 10, start_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(image, "G", (goal_x - 10, goal_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        cv2.imwrite(self.image_path, image)
        print(f"✓ Created default scene: {self.image_path}")
    
    def start_stream(self):
        """Start providing frames (just sets flag)."""
        self.is_streaming = True
        print("✓ Dynamic image provider started")
        print("  Tip: Run 'python dynamic_scene_generator.py server' in another terminal")
        print("       to update the image automatically!")
    
    def get_frame(self):
        """
        Get current frame by RELOADING from file.
        This detects changes made by external processes!
        
        Returns:
            numpy.ndarray: BGR image (reloaded from file)
        """
        if not self.is_streaming:
            return None
        
        try:
            # Check if file was modified
            current_modified = os.path.getmtime(self.image_path)
            
            # Always reload (or reload only if changed)
            # For dynamic testing, ALWAYS reload to catch changes
            image = cv2.imread(self.image_path)
            
            if image is not None:
                with self.frame_lock:
                    self.current_image = image
                
                # Update FPS counter
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_frame_time = current_time
                
                # Detect if file changed
                if current_modified > self.last_modified:
                    print(f"  📸 IMAGE UPDATED! (New timestamp: {current_modified})")
                    self.last_modified = current_modified
                
                return image.copy()
            
            return None
            
        except Exception as e:
            print(f"⚠ Error reading image: {str(e)}")
            return self.current_image.copy() if self.current_image is not None else None
    
    def get_fps(self):
        """Get current FPS."""
        return self.fps
    
    def stop_stream(self):
        """Stop the stream."""
        self.is_streaming = False
        print("✓ Stopped")
    
    def disconnect(self):
        """Cleanup."""
        self.stop_stream()
        print("✓ Dynamic image input disconnected")
    
    # Compatibility methods (for DroneInput interface)
    def takeoff(self):
        """Dummy method for compatibility."""
        pass
    
    def land(self):
        """Dummy method for compatibility."""
        pass
    
    def emergency_stop(self):
        """Dummy method for compatibility."""
        pass


# Test the dynamic input
if __name__ == "__main__":
    print("Testing Dynamic Image Input")
    print("="*60)
    print("\nThis will watch for changes to the image file.")
    print("Try running this in another terminal to update the image:")
    print("  python dynamic_scene_generator.py server")
    print("="*60 + "\n")
    
    # Create input
    img_input = DynamicImageInput("data/test_images/current_scene.jpg")
    
    # Connect
    if img_input.connect():
        # Start streaming
        img_input.start_stream()
        
        print("\nReading frames for 30 seconds...")
        print("If you update the image file, you'll see a notification!")
        print("Press 'Q' to quit early\n")
        
        start_time = time.time()
        
        while time.time() - start_time < 30:
            frame = img_input.get_frame()
            
            if frame is not None:
                # Show frame
                cv2.putText(frame, f"FPS: {img_input.get_fps():.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Watching for changes...", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("Dynamic Image Input Test", frame)
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
        
        cv2.destroyAllWindows()
        img_input.disconnect()
        
        print("\n✓ Test complete!")
    else:
        print("✗ Failed to connect")
