"""
Test script to verify coordinate system fix
"""

import numpy as np

def test_movement_system():
    """Test the fixed movement system."""
    
    print("="*70)
    print("TESTING COORDINATE SYSTEM FIX")
    print("="*70)
    
    # Image coordinate system mapping
    heading_map = {
        0: (1, 0),      # East: right
        45: (1, 1),     # Southeast: down-right
        90: (0, 1),     # South: down
        135: (-1, 1),   # Southwest: down-left
        180: (-1, 0),   # West: left
        225: (-1, -1),  # Northwest: up-left
        270: (0, -1),   # North: up
        315: (1, -1)    # Northeast: up-right
    }
    
    # Test moving from (10, 10) toward (90, 90)
    start = (10, 10)
    goal = (90, 90)
    
    print(f"\nGoal: Move from {start} to {goal}")
    print(f"Need to go: RIGHT +80, DOWN +80")
    print()
    
    # Test each heading
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        dx_unit, dy_unit = heading_map[angle]
        
        # Move 10 cells
        distance = 10
        dx = distance * dx_unit
        dy = distance * dy_unit
        
        new_pos = (start[0] + dx, start[1] + dy)
        
        # Check if moving toward goal
        old_dist = np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)
        new_dist = np.sqrt((goal[0]-new_pos[0])**2 + (goal[1]-new_pos[1])**2)
        closer = "✓ CLOSER" if new_dist < old_dist else "✗ farther"
        
        print(f"{angle:3d}°: move({distance}) → ({dx:+3d}, {dy:+3d}) = {new_pos} {closer}")
    
    print("\n" + "="*70)
    print("EXPECTED RESULT:")
    print("  45° (Southeast) should move CLOSER ✓")
    print("  This means: +X (right) and +Y (down)")
    print("="*70)
    
    # Test the actual path from (10,10) to (90,90)
    print("\nTESTING ACTUAL PATH:")
    pos = list(start)
    heading = 45  # Southeast
    
    print(f"Start: {tuple(pos)}")
    
    for i in range(8):
        dx_unit, dy_unit = heading_map[heading]
        distance = 10
        dx = distance * dx_unit
        dy = distance * dy_unit
        
        pos[0] += dx
        pos[1] += dy
        
        dist_to_goal = np.sqrt((goal[0]-pos[0])**2 + (goal[1]-pos[1])**2)
        print(f"Step {i+1}: {tuple(pos)} (distance to goal: {dist_to_goal:.1f})")
    
    print(f"\nFinal: {tuple(pos)}")
    print(f"Goal:  {goal}")
    print(f"Match: {tuple(pos) == goal}")

if __name__ == "__main__":
    test_movement_system()
