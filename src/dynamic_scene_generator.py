"""
Dynamic Image Generator - Creates changing disaster scenes
Simulates a dynamic environment where obstacles appear/disappear over time
"""

import cv2
import numpy as np
import time
import os
import random
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
        
        # Safe zones (keep these clear for start/goal)
        self.safe_zones = [
            {'pos': self.start_pos, 'radius': 80},
            {'pos': self.goal_pos, 'radius': 80}
        ]
        
        # Dynamic obstacle schedule
        self.dynamic_events = self._create_event_schedule()
        
        # Background texture for realism
        self.background = self._generate_disaster_background()
        
        print(f"Dynamic Scene Generator initialized")
        print(f"  Resolution: {width}x{height}")
        print(f"  Output: {output_path}")
        print(f"  Dynamic events scheduled: {len(self.dynamic_events)}")
    
    def _generate_disaster_background(self):
        """
        Generate a realistic disaster scene background.
        Includes dirt, cracks, debris patterns.
        """
        # Start with brownish dirt color
        background = np.ones((self.height, self.width, 3), dtype=np.uint8)
        
        # Base: dirt/sandy ground color
        base_color = np.array([150, 180, 200])  # Light sandy brown (BGR)
        background[:, :] = base_color
        
        # Add random dirt variations
        noise = np.random.randint(-30, 30, (self.height, self.width, 3), dtype=np.int16)
        background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Blur to make it look more natural
        background = cv2.GaussianBlur(background, (15, 15), 0)
        
        # Add some darker patches (dirt, shadows)
        num_patches = random.randint(5, 10)
        for _ in range(num_patches):
            x = random.randint(0, self.width - 200)
            y = random.randint(0, self.height - 200)
            w = random.randint(100, 300)
            h = random.randint(100, 300)
            
            # Dark patch
            patch_color = np.array([100, 120, 130])  # Darker dirt
            cv2.ellipse(background, (x + w//2, y + h//2), (w//2, h//2), 
                       0, 0, 360, patch_color.tolist(), -1)
        
        # Blur again for natural blending
        background = cv2.GaussianBlur(background, (21, 21), 0)
        
        # Add some cracks/lines
        num_cracks = random.randint(8, 15)
        for _ in range(num_cracks):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = x1 + random.randint(-200, 200)
            y2 = y1 + random.randint(-200, 200)
            
            thickness = random.randint(1, 3)
            crack_color = (80, 90, 100)  # Dark crack color
            cv2.line(background, (x1, y1), (x2, y2), crack_color, thickness)
        
        return background
    
    def _create_event_schedule(self):
        """
        Create a schedule of events that will happen.
        Uses random positions that don't overlap.
        
        Returns:
            list: List of (iteration, event_type, params) tuples
        """
        events = []
        
        # Initial obstacles (iteration 0) - random non-overlapping
        num_initial_obstacles = random.randint(4, 8)
        for i in range(num_initial_obstacles):
            pos, size = self._get_random_obstacle_placement()
            if pos is not None:
                events.append((0, 'add_obstacle', {'pos': pos, 'size': size}))
        
        # Add obstacles appearing over time
        for iteration in [5, 10, 15, 20, 25]:
            if random.random() < 0.7:  # 70% chance of new obstacle
                events.append((iteration, 'add_obstacle', {
                    'pos': None,  # Will be calculated when event triggers
                    'size': None,
                    'random': True,
                    'label': 'NEW!' if random.random() < 0.5 else None
                }))
            
            if random.random() < 0.3 and iteration > 5:  # 30% chance to remove obstacle
                # Will pick random obstacle to remove
                events.append((iteration, 'remove_random', {}))
        
        return events
    
    def _get_random_obstacle_placement(self, min_size=60, max_size=120):
        """
        Find a random position and size for an obstacle that doesn't overlap
        with existing obstacles or safe zones.
        
        Returns:
            tuple: ((x, y), (w, h)) or (None, None) if no valid position found
        """
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # Random size
            w = random.randint(min_size, max_size)
            h = random.randint(min_size, max_size)
            
            # Random position (with margin from edges)
            margin = 50
            x = random.randint(margin, self.width - w - margin)
            y = random.randint(margin, self.height - h - margin)
            
            # Check if this position is valid
            if self._is_position_valid(x, y, w, h):
                return (x, y), (w, h)
        
        # Couldn't find valid position
        print(f"  ⚠ Could not place obstacle after {max_attempts} attempts")
        return None, None
    
    def _is_position_valid(self, x, y, w, h):
        """
        Check if a position is valid (doesn't overlap with obstacles or safe zones).
        
        Args:
            x, y: Top-left position
            w, h: Size
            
        Returns:
            bool: True if valid position
        """
        # Check overlap with safe zones
        center_x = x + w // 2
        center_y = y + h // 2
        
        for safe_zone in self.safe_zones:
            sx, sy = safe_zone['pos']
            radius = safe_zone['radius']
            
            dist = np.sqrt((center_x - sx)**2 + (center_y - sy)**2)
            if dist < radius + max(w, h) // 2:
                return False
        
        # Check overlap with existing obstacles (with buffer)
        buffer = 20  # Minimum gap between obstacles
        for obs in self.obstacles:
            ox, oy = obs['pos']
            ow, oh = obs['size']
            
            # Check if rectangles overlap (with buffer)
            if not (x + w + buffer < ox or 
                    x > ox + ow + buffer or 
                    y + h + buffer < oy or 
                    y > oy + oh + buffer):
                return False
        
        return True
    
    def generate_scene(self):
        """
        Generate the current scene based on iteration count.
        Processes scheduled events and creates the image.
        
        Returns:
            numpy.ndarray: Generated scene image
        """
        # Process events for this iteration
        self._process_events()
        
        # Start with background
        image = self.background.copy()
        
        # Draw obstacles (dark debris with varied appearance)
        for i, obs in enumerate(self.obstacles):
            x, y = obs['pos']
            w, h = obs['size']
            
            # Random obstacle appearance
            obstacle_type = random.choice(['dark', 'concrete', 'debris'])
            
            if obstacle_type == 'dark':
                # Very dark obstacle
                color = (random.randint(10, 30), random.randint(10, 30), random.randint(10, 30))
                cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
                # Dark red border
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 100), 3)
            
            elif obstacle_type == 'concrete':
                # Gray concrete-like
                gray = random.randint(60, 90)
                color = (gray, gray, gray)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
                # Add some texture
                for _ in range(10):
                    tx = random.randint(x, x + w)
                    ty = random.randint(y, y + h)
                    cv2.circle(image, (tx, ty), 2, (gray - 20, gray - 20, gray - 20), -1)
                cv2.rectangle(image, (x, y), (x + w, y + h), (40, 40, 40), 2)
            
            else:  # debris
                # Brownish debris pile
                color = (random.randint(80, 120), random.randint(100, 140), random.randint(90, 130))
                # Irregular shape
                pts = np.array([
                    [x, y + h//3],
                    [x + w//3, y],
                    [x + 2*w//3, y + h//4],
                    [x + w, y + h//2],
                    [x + 2*w//3, y + h],
                    [x + w//3, y + 3*h//4]
                ])
                cv2.fillPoly(image, [pts], color)
                cv2.polylines(image, [pts], True, (60, 80, 70), 2)
            
            # Add label if present
            if 'label' in obs and obs['label']:
                cv2.putText(image, obs['label'], (x + 5, y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Mark start position (white circle)
        cv2.circle(image, self.start_pos, 40, (255, 255, 255), -1)
        cv2.circle(image, self.start_pos, 40, (200, 200, 200), 2)
        
        # Mark goal position (white circle)
        cv2.circle(image, self.goal_pos, 40, (255, 255, 255), -1)
        cv2.circle(image, self.goal_pos, 40, (200, 200, 200), 2)
        
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
                if params.get('random', False):
                    # Generate random position
                    pos, size = self._get_random_obstacle_placement()
                    if pos is not None:
                        params['pos'] = pos
                        params['size'] = size
                        self._add_obstacle(params)
                        print(f"  [Event] Added random obstacle at iteration {self.iteration}")
                else:
                    self._add_obstacle(params)
                    print(f"  [Event] Added obstacle at iteration {self.iteration}")
            
            elif event_type == 'remove_obstacle':
                self._remove_obstacle(params['index'])
                print(f"  [Event] Removed obstacle at iteration {self.iteration}")
            
            elif event_type == 'remove_random':
                if len(self.obstacles) > 0:
                    index = random.randint(0, len(self.obstacles) - 1)
                    self._remove_obstacle(index)
                    print(f"  [Event] Removed random obstacle at iteration {self.iteration}")
            
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
        self.background = self._generate_disaster_background()


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
            time.sleep(8)  # Update every 2 seconds
    
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