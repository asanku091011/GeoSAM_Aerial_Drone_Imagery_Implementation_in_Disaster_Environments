"""
FIXED Path Converter for Image Coordinates
This replaces the version that was giving wrong angles
"""

import numpy as np


class PathConverter:
    """
    Converts paths to robot commands using IMAGE coordinate system.
    
    IMAGE COORDINATES:
    (0,0) at top-left, +X = right, +Y = down
    
    HEADING ANGLES:
    0°=East, 45°=Southeast, 90°=South, 135°=Southwest,
    180°=West, 225°=Northwest, 270°=North, 315°=Northeast
    """
    
    def __init__(self, map_builder, unit_scale=1.0):
        self.map_builder = map_builder
        self.unit_scale = unit_scale
        print(f"Path Converter initialized (IMAGE coordinates)")
    
    def convert_path_to_commands(self, path, start_heading=0):
        """Convert path to turn() and move() commands."""
        if not path or len(path) < 2:
            return []
        
        commands = []
        current_heading = start_heading
        i = 0
        
        while i < len(path) - 1:
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            
            target_heading = self._calculate_heading(dx, dy)
            if target_heading is None:
                i += 1
                continue
            
            if target_heading != current_heading:
                commands.append(f"turn({target_heading})")
                current_heading = target_heading
            
            # Count consecutive moves in same direction
            distance = 1
            j = i + 1
            while j < len(path) - 1:
                next_dx = path[j+1][0] - path[j][0]
                next_dy = path[j+1][1] - path[j][1]
                if self._calculate_heading(next_dx, next_dy) == current_heading:
                    distance += 1
                    j += 1
                else:
                    break
            
            commands.append(f"move({int(distance * self.unit_scale)})")
            i += distance
        
        print(f"✓ Converted path to {len(commands)} commands")
        return commands
    
    def _calculate_heading(self, dx, dy):
        """Calculate heading from displacement (IMAGE coordinates)."""
        if dx == 0 and dy == 0:
            return None
        elif dx > 0 and dy == 0:
            return 0    # East
        elif dx > 0 and dy > 0:
            return 45   # Southeast ← THIS IS THE FIX!
        elif dx == 0 and dy > 0:
            return 90   # South
        elif dx < 0 and dy > 0:
            return 135  # Southwest
        elif dx < 0 and dy == 0:
            return 180  # West
        elif dx < 0 and dy < 0:
            return 225  # Northwest
        elif dx == 0 and dy < 0:
            return 270  # North
        elif dx > 0 and dy < 0:
            return 315  # Northeast
        return None
    
    def optimize_commands(self, commands):
        """Combine consecutive moves."""
        if not commands:
            return []
        optimized = []
        i = 0
        while i < len(commands):
            if commands[i].startswith('move('):
                total = int(commands[i][5:-1])
                j = i + 1
                while j < len(commands) and commands[j].startswith('move('):
                    total += int(commands[j][5:-1])
                    j += 1
                optimized.append(f'move({total})')
                i = j
            else:
                optimized.append(commands[i])
                i += 1
        return optimized
    
    def save_commands(self, commands, filename="outputs/robot_path.txt"):
        """Save to file."""
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                for cmd in commands:
                    f.write(cmd + '\n')
            return True
        except:
            return False
    
    def preview_commands(self, commands, max_lines=10):
        """Print preview."""
        print(f"\nCommands:")
        for i, cmd in enumerate(commands[:max_lines]):
            print(f"  {i+1}. {cmd}")
        if len(commands) > max_lines:
            print(f"  ... +{len(commands)-max_lines} more")
