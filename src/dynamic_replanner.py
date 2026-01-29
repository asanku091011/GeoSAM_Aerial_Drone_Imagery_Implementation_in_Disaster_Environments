"""
Dynamic replanner that detects environmental changes and triggers path updates.
Monitors the navigation map and current path for obstacles and replans when needed.
"""

import numpy as np
import time
import config

class DynamicReplanner:
    """
    Manages dynamic replanning when environment changes.
    
    This class:
    - Detects when new obstacles appear in the map
    - Checks if current path is still valid
    - Triggers replanning when necessary
    - Prevents excessive replanning with timing controls
    """
    
    def __init__(self, map_builder, planner):
        """
        Initialize dynamic replanner.
        
        Args:
            map_builder: MapBuilder instance
            planner: Path planning algorithm instance (A*, RRT*, or Greedy)
        """
        self.map_builder = map_builder
        self.planner = planner
        
        # State tracking
        self.current_path = None
        self.current_position = None
        self.goal_position = None
        self.last_replan_time = 0
        
        # Statistics
        self.replan_count = 0
        self.collision_detections = 0
        self.map_change_detections = 0
        
        print("Dynamic Replanner initialized")
    
    def set_path(self, path, start, goal):
        """
        Set the current path being followed.
        
        Args:
            path (list): Current path as list of (x, y) tuples
            start (tuple): Start position
            goal (tuple): Goal position
        """
        self.current_path = path
        self.current_position = start
        self.goal_position = goal
    
    def update_position(self, position):
        """
        Update the robot's current position.
        
        Args:
            position (tuple): Current (x, y) position in grid coordinates
        """
        self.current_position = position
    
    def check_and_replan(self):
        """
        Check if replanning is needed and trigger it if necessary.
        
        Returns:
            tuple: (needs_replan, new_path, reason)
                - needs_replan (bool): Whether replanning occurred
                - new_path (list): New path if replanned, None otherwise
                - reason (str): Reason for replanning
        """
        # Check if we have a path to monitor
        if self.current_path is None or self.current_position is None:
            return False, None, "No active path"
        
        # Check minimum replan interval
        current_time = time.time()
        time_since_last_replan = current_time - self.last_replan_time
        
        if time_since_last_replan < config.MIN_REPLAN_INTERVAL:
            return False, None, f"Too soon (wait {config.MIN_REPLAN_INTERVAL - time_since_last_replan:.1f}s)"
        
        # Check for various replanning triggers
        
        # 1. Check if significant map changes occurred
        map_change_pct = self.map_builder.calculate_change_percentage()
        if map_change_pct > config.MAP_UPDATE_THRESHOLD * 100:
            self.map_change_detections += 1
            return self._trigger_replan("Map changed significantly ({:.1f}%)".format(map_change_pct))
        
        # 2. Check if current path has obstacles
        if self._path_has_collision():
            self.collision_detections += 1
            return self._trigger_replan("Obstacle detected on path")
        
        # 3. Check if we can no longer reach the goal
        if not self.map_builder.is_cell_free(self.goal_position[0], self.goal_position[1]):
            return self._trigger_replan("Goal is now blocked")
        
        # No replanning needed
        return False, None, "Path is clear"
    
    def _path_has_collision(self):
        """
        Check if the current path intersects with obstacles.
        Only checks a few waypoints ahead for efficiency.
        
        Returns:
            bool: True if collision detected on path
        """
        if not self.current_path:
            return False
        
        # Find current waypoint index
        current_idx = self._find_closest_waypoint_index(self.current_position)
        
        # Check next N waypoints
        check_ahead = min(config.COLLISION_CHECK_AHEAD, 
                         len(self.current_path) - current_idx)
        
        for i in range(current_idx, current_idx + check_ahead):
            if i >= len(self.current_path):
                break
            
            waypoint = self.current_path[i]
            
            # Check if waypoint is now an obstacle
            if not self.map_builder.is_cell_free(waypoint[0], waypoint[1]):
                return True
            
            # Check path segment to next waypoint
            if i < len(self.current_path) - 1:
                next_waypoint = self.current_path[i + 1]
                if not self._is_path_segment_free(waypoint, next_waypoint):
                    return True
        
        return False
    
    def _is_path_segment_free(self, start, end):
        """
        Check if a straight path segment is collision-free.
        
        Args:
            start (tuple): Start position
            end (tuple): End position
            
        Returns:
            bool: True if segment is free
        """
        # Bresenham's line algorithm
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            if not self.map_builder.is_cell_free(x, y):
                return False
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True
    
    def _find_closest_waypoint_index(self, position):
        """
        Find the index of the closest waypoint to current position.
        
        Args:
            position (tuple): Current position
            
        Returns:
            int: Index of closest waypoint
        """
        if not self.current_path:
            return 0
        
        min_distance = float('inf')
        closest_idx = 0
        
        for i, waypoint in enumerate(self.current_path):
            dx = waypoint[0] - position[0]
            dy = waypoint[1] - position[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        return closest_idx
    
    def _trigger_replan(self, reason):
        """
        Trigger a replan operation.
        
        Args:
            reason (str): Reason for replanning
            
        Returns:
            tuple: (True, new_path, reason)
        """
        print(f"🔄 Replanning triggered: {reason}")
        
        # Plan new path from current position to goal
        new_path = self.planner.plan(self.current_position, self.goal_position)
        
        if new_path:
            self.current_path = new_path
            self.last_replan_time = time.time()
            self.replan_count += 1
            print(f"✓ New path found with {len(new_path)} waypoints")
            return True, new_path, reason
        else:
            print(f"✗ Replanning failed - no path found!")
            return True, None, reason + " (replan failed)"
    
    def force_replan(self):
        """
        Force an immediate replan regardless of timing constraints.
        
        Returns:
            list: New path, or None if planning failed
        """
        print("🔄 Forced replanning...")
        self.last_replan_time = 0  # Reset timer
        _, new_path, _ = self.check_and_replan()
        return new_path
    
    def get_statistics(self):
        """
        Get replanning statistics.
        
        Returns:
            dict: Statistics about replanning operations
        """
        return {
            'replan_count': self.replan_count,
            'collision_detections': self.collision_detections,
            'map_change_detections': self.map_change_detections,
            'has_active_path': self.current_path is not None,
            'current_path_length': len(self.current_path) if self.current_path else 0
        }
    
    def reset(self):
        """Reset the replanner state."""
        self.current_path = None
        self.current_position = None
        self.goal_position = None
        self.last_replan_time = 0
        print("Dynamic replanner reset")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Dynamic Replanner Module")
    print("=" * 50)
    
    from map_builder import MapBuilder
    from astar import AStarPlanner
    import cv2
    
    # Create initial map
    map_builder = MapBuilder(width=50, height=50)
    grid = np.ones((50, 50), dtype=np.uint8)
    grid[20:22, 15:35] = 0  # Initial obstacle
    map_builder.update_from_segmentation(grid)
    
    # Create planner and replanner
    planner = AStarPlanner(map_builder)
    replanner = DynamicReplanner(map_builder, planner)
    
    # Plan initial path
    start = (5, 25)
    goal = (45, 25)
    
    print("\nPlanning initial path...")
    path = planner.plan(start, goal)
    
    if path:
        print(f"✓ Initial path: {len(path)} waypoints")
        replanner.set_path(path, start, goal)
        
        # Visualize initial path
        vis1 = map_builder.visualize(path=path, start=start, goal=goal)
        cv2.imshow("Initial Path", vis1)
        
        # Simulate robot moving and new obstacle appearing
        print("\n--- Simulating navigation ---")
        replanner.update_position((15, 25))
        
        # Add new obstacle on the path!
        print("\n⚠ New obstacle detected!")
        grid[24:26, 25:30] = 0
        map_builder.update_from_segmentation(grid)
        
        # Check if replanning needed
        needs_replan, new_path, reason = replanner.check_and_replan()
        
        if needs_replan:
            print(f"\nReplanning result: {reason}")
            
            if new_path:
                print(f"✓ New path: {len(new_path)} waypoints")
                
                # Visualize new path
                vis2 = map_builder.visualize(path=new_path, start=(15, 25), goal=goal)
                cv2.imshow("Replanned Path", vis2)
        
        # Show statistics
        stats = replanner.get_statistics()
        print(f"\n--- Replanner Statistics ---")
        print(f"Total replans: {stats['replan_count']}")
        print(f"Collision detections: {stats['collision_detections']}")
        print(f"Map change detections: {stats['map_change_detections']}")
        
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        print("✗ Failed to plan initial path")