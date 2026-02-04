"""
Dynamic Image Generator - Creates changing disaster scenes
Simulates a dynamic environment where obstacles appear/disappear over time
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path


class DynamicSceneGenerator:
    """
    Generates disaster scene images that change over time.
    
    Simulates dynamic scenarios:
    - New obstacles appearing (debris falling)
    - Obstacles disappearing (cleared paths)
    - Moving obstacles
    - Time-based changes
    """
    
    def __init__(self, width=960, height=720, output_path="data/test_images/current_scene.jpg"):
        """
        Initialize the dynamic scene generator.
        
        Args:
            width (int): Image width
            height (int): Image height
            output_path (str): Where to save the current scene
        """
        self.width = width
        self.height = height
        self.output_path = output_path
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Scene state
        self.iteration = 0
        self.obstacles = []
        self.start_pos = (int(width * 0.10), int(height * 0.10))
        self.goal_pos = (int(width * 0.90), int(height * 0.90))
        
        # Dynamic obstacle schedule
        self.dynamic_events = self._create_event_schedule()
        
        print(f"Dynamic Scene Generator initialized")
        print(f"  Resolution: {width}x{height}")
        print(f"  Output: {output_path}")
        print(f"  Dynamic events scheduled: {len(self.dynamic_events)}")
    
    def _create_event_schedule(self):
        """
        Create a schedule of events that will happen.
        
        Returns:
            list: List of (iteration, event_type, params) tuples
        """
        events = [
            # Format: (iteration, event_type, params)
            
            # Initial obstacles (iteration 0)
            (0, 'add_obstacle', {'pos': (300, 200), 'size': (80, 100)}),
            (0, 'add_obstacle', {'pos': (450, 400), 'size': (90, 90)}),
            (0, 'add_obstacle', {'pos': (600, 300), 'size': (75, 110)}),
            (0, 'add_obstacle', {'pos': (450, 200), 'size': (40, 40)}),
            (0, 'add_obstacle', {'pos': (500, 200), 'size': (40, 40)}),
            (0, 'add_obstacle', {'pos': (550, 200), 'size': (40, 40)}),
            (0, 'add_obstacle', {'pos': (600, 200), 'size': (40, 40)}),
            (0, 'add_obstacle', {'pos': (650, 200), 'size': (40, 40)}),
            (0, 'add_obstacle', {'pos': (280, 320), 'size': (40, 40)}),
            (0, 'add_obstacle', {'pos': (280, 380), 'size': (40, 40)}),
            (0, 'add_obstacle', {'pos': (280, 440), 'size': (40, 40)}),
            (0, 'add_obstacle', {'pos': (280, 500), 'size': (40, 40)}),
            (0, 'add_obstacle', {'pos': (280, 560), 'size': (40, 40)}),
            

            
            # New obstacle appears at iteration 5
            (5, 'add_obstacle', {'pos': (350, 350), 'size': (100, 100), 'label': 'NEW!'}),
            
            # Clear a path at iteration 10
            (10, 'remove_obstacle', {'index': 1}),
            
            # Add another obstacle at iteration 15
            (15, 'add_obstacle', {'pos': (500, 250), 'size': (85, 95), 'label': 'DEBRIS'}),
            
            # Move an obstacle at iteration 20
            (20, 'move_obstacle', {'index': 0, 'new_pos': (320, 220)}),
            
            # More obstacles appear at iteration 25
            (25, 'add_obstacle', {'pos': (700, 500), 'size': (70, 80)}),
        ]
        
        return events
    
    def generate_scene(self):
        """
        Generate the current scene based on iteration count.
        Processes scheduled events and creates the image.
        
        Returns:
            numpy.ndarray: Generated scene image
        """
        # Process events for this iteration
        self._process_events()
        
        # Create base image (bright green = safe terrain)
        image = np.ones((self.height, self.width, 3), dtype=np.uint8)
        image[:, :] = (255, 255, 255)  # Bright green
        
        # Draw obstacles (very dark with red borders)
        for i, obs in enumerate(self.obstacles):
            x, y = obs['pos']
            w, h = obs['size']
            
            # Draw obstacle (dark black)
            cv2.rectangle(image, (x, y), (x + w, y + h), (15, 15, 15), -1)
            
            # Red border to make it obvious
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            
            # Add label if present
            if 'label' in obs:
                cv2.putText(image, obs['label'], (x + 5, y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Mark start position (subtle)
        cv2.circle(image, self.start_pos, 40, (255, 255, 255), -1)
        #cv2.putText(image, "S", (self.start_pos[0] - 10, self.start_pos[1] + 5),
                   #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
        
        # Mark goal position (subtle)
        cv2.circle(image, self.goal_pos, 40, (255, 255, 255), -1)
        #cv2.putText(image, "G", (self.goal_pos[0] - 10, self.goal_pos[1] + 5),
                   #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
        
        # Add iteration counter
        cv2.putText(image, f"Iteration {self.iteration}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add obstacle count
        cv2.putText(image, f"Obstacles: {len(self.obstacles)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        self.iteration += 1
        return image
    
    def _process_events(self):
        """Process scheduled events for current iteration."""
        current_events = [e for e in self.dynamic_events if e[0] == self.iteration]
        
        for event in current_events:
            _, event_type, params = event
            
            if event_type == 'add_obstacle':
                self._add_obstacle(params)
                print(f"  [Event] Added obstacle at iteration {self.iteration}")
            
            elif event_type == 'remove_obstacle':
                self._remove_obstacle(params['index'])
                print(f"  [Event] Removed obstacle at iteration {self.iteration}")
            
            elif event_type == 'move_obstacle':
                self._move_obstacle(params['index'], params['new_pos'])
                print(f"  [Event] Moved obstacle at iteration {self.iteration}")
    
    def _add_obstacle(self, params):
        """Add a new obstacle."""
        self.obstacles.append(params)
    
    def _remove_obstacle(self, index):
        """Remove an obstacle by index."""
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)
    
    def _move_obstacle(self, index, new_pos):
        """Move an obstacle to a new position."""
        if 0 <= index < len(self.obstacles):
            self.obstacles[index]['pos'] = new_pos
    
    def save_current_scene(self):
        """
        Generate and save the current scene to file.
        This is what ImageInput will read.
        """
        image = self.generate_scene()
        cv2.imwrite(self.output_path, image)
        return image
    
    def reset(self):
        """Reset the scene to initial state."""
        self.iteration = 0
        self.obstacles = []


def run_dynamic_scene_server():
    """
    Run a background server that continuously updates the scene image.
    This runs in parallel with your navigation system.
    """
    print("="*70)
    print("🎬 DYNAMIC SCENE SERVER")
    print("="*70)
    print("\nThis will continuously update the test image to simulate")
    print("a changing environment while your navigation system runs.")
    print("\nPress Ctrl+C to stop")
    print("="*70)
    
    generator = DynamicSceneGenerator()
    
    try:
        while True:
            # Generate and save new scene
            image = generator.save_current_scene()
            
            print(f"\nIteration {generator.iteration - 1}:")
            print(f"  Obstacles: {len(generator.obstacles)}")
            print(f"  Saved to: {generator.output_path}")
            
            # Show preview (optional)
            cv2.imshow("Dynamic Scene (Press Q to hide)", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
            
            # Wait before next update
            # Match this to your navigation system's iteration speed
            time.sleep(2)  # Update every 2 seconds
    
    except KeyboardInterrupt:
        print("\n\n⚠ Stopped by user")
        cv2.destroyAllWindows()


def create_test_sequence():
    """
    Create a sequence of test images and save them.
    Useful for testing without running the server.
    """
    print("Creating test sequence...")
    
    generator = DynamicSceneGenerator()
    output_dir = "data/test_images/sequence"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 30 frames
    for i in range(30):
        image = generator.generate_scene()
        filename = os.path.join(output_dir, f"scene_{i:03d}.jpg")
        cv2.imwrite(filename, image)
        print(f"  Created: {filename}")
    
    print(f"\n✓ Created {30} test images in {output_dir}")
    print("You can manually copy these to data/test_images/current_scene.jpg")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run dynamic server
        run_dynamic_scene_server()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "sequence":
        # Create test sequence
        create_test_sequence()
    
    else:
        # Default: create one scene and show it
        print("Dynamic Scene Generator")
        print("="*50)
        print("\nOptions:")
        print("  python dynamic_scene.py server    - Run live server")
        print("  python dynamic_scene.py sequence  - Create test sequence")
        print("  python dynamic_scene.py           - Create single scene")
        print()
        
        generator = DynamicSceneGenerator()
        
        for i in range(10):
            image = generator.save_current_scene()
            print(f"Iteration {i}: {len(generator.obstacles)} obstacles")
            
            cv2.imshow("Scene", image)
            key = cv2.waitKey(500)
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("\n✓ Demo complete")
