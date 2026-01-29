"""
Greedy best-first search path planning algorithm.
Greedy search is fast but not guaranteed to find the optimal path.
It always moves toward the goal, which makes it very efficient in open spaces.
"""

import numpy as np
import heapq
import config

class GreedyPlanner:
    """
    Greedy best-first search pathfinding algorithm.
    
    Greedy search always expands the node that appears closest to the goal,
    based on a heuristic function. It's faster than A* but doesn't guarantee
    the optimal path.
    
    Advantages:
    - Very fast in favorable conditions
    - Low memory usage
    - Good for real-time applications
    
    Disadvantages:
    - Can get stuck in local minima
    - Not guaranteed to find optimal path
    - May fail where A* succeeds
    """
    
    def __init__(self, map_builder):
        """
        Initialize Greedy planner.
        
        Args:
            map_builder: MapBuilder instance containing the navigation grid
        """
        self.map_builder = map_builder
        self.max_iterations = config.GREEDY_MAX_ITERATIONS
        
        print(f"Greedy Planner initialized (max_iter: {self.max_iterations})")
    
    def plan(self, start, goal):
        """
        Plan a path from start to goal using greedy best-first search.
        
        Args:
            start (tuple): Start position (x, y) in grid coordinates
            goal (tuple): Goal position (x, y) in grid coordinates
            
        Returns:
            list: Path as list of (x, y) tuples, or None if no path exists
        """
        # Validate inputs
        if not self._validate_positions(start, goal):
            return None
        
        # Initialize data structures
        open_set = []  # Priority queue: (heuristic, counter, node)
        heapq.heappush(open_set, (self._heuristic(start, goal), 0, start))
        
        came_from = {}  # For reconstructing path
        visited = set()  # Closed set
        counter = 1
        iterations = 0
        
        # Greedy search main loop
        while open_set and iterations < self.max_iterations:
            iterations += 1
            
            # Get node with lowest heuristic (closest to goal)
            _, _, current = heapq.heappop(open_set)
            
            # Goal reached!
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return path
            
            # Skip if already visited
            if current in visited:
                continue
            
            visited.add(current)
            
            # Explore neighbors
            neighbors = self.map_builder.get_neighbors(
                current[0], current[1], 
                allow_diagonal=True
            )
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                # Record path
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    h = self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (h, counter, neighbor))
                    counter += 1
        
        # Check if we reached max iterations
        if iterations >= self.max_iterations:
            print(f"⚠ Greedy: Max iterations ({self.max_iterations}) reached")
        else:
            print("⚠ Greedy: No path found to goal")
        
        return None
    
    def plan_with_potential_field(self, start, goal):
        """
        Alternative greedy approach using artificial potential fields.
        Treats goal as attractive force and obstacles as repulsive forces.
        
        Args:
            start (tuple): Start position (x, y)
            goal (tuple): Goal position (x, y)
            
        Returns:
            list: Path as list of (x, y) tuples, or None if failed
        """
        if not self._validate_positions(start, goal):
            return None
        
        path = [start]
        current = start
        visited = set([start])
        
        max_steps = self.max_iterations
        stuck_threshold = 5  # If we don't make progress, we're stuck
        stuck_count = 0
        last_distance = self._heuristic(start, goal)
        
        for step in range(max_steps):
            # Check if reached goal
            if current == goal:
                return path
            
            # Get all free neighbors
            neighbors = self.map_builder.get_neighbors(
                current[0], current[1],
                allow_diagonal=True
            )
            
            if not neighbors:
                print("⚠ Greedy PF: Trapped with no free neighbors")
                return None
            
            # Calculate potential for each neighbor
            best_neighbor = None
            best_potential = float('inf')
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                # Attractive potential (distance to goal)
                attractive = self._heuristic(neighbor, goal)
                
                # Repulsive potential (distance from obstacles)
                repulsive = self._calculate_repulsive_potential(neighbor)
                
                # Total potential
                total_potential = attractive - repulsive * 0.5
                
                if total_potential < best_potential:
                    best_potential = total_potential
                    best_neighbor = neighbor
            
            # If no unvisited neighbors, try visited ones (backtrack)
            if best_neighbor is None:
                for neighbor in neighbors:
                    dist = self._heuristic(neighbor, goal)
                    if dist < best_potential:
                        best_potential = dist
                        best_neighbor = neighbor
            
            # Still no neighbor? We're trapped
            if best_neighbor is None:
                print("⚠ Greedy PF: No valid moves available")
                return None
            
            # Move to best neighbor
            current = best_neighbor
            path.append(current)
            visited.add(current)
            
            # Check if we're making progress
            current_distance = self._heuristic(current, goal)
            if current_distance >= last_distance:
                stuck_count += 1
                if stuck_count >= stuck_threshold:
                    print("⚠ Greedy PF: Stuck in local minimum")
                    return None
            else:
                stuck_count = 0
            
            last_distance = current_distance
        
        print(f"⚠ Greedy PF: Max steps ({max_steps}) reached")
        return None
    
    def _validate_positions(self, start, goal):
        """Validate start and goal positions."""
        if not self.map_builder.is_valid_cell(start[0], start[1]):
            print(f"✗ Greedy: Start position {start} is out of bounds")
            return False
        
        if not self.map_builder.is_valid_cell(goal[0], goal[1]):
            print(f"✗ Greedy: Goal position {goal} is out of bounds")
            return False
        
        if not self.map_builder.is_cell_free(start[0], start[1]):
            print(f"✗ Greedy: Start position {start} is in obstacle")
            return False
        
        if not self.map_builder.is_cell_free(goal[0], goal[1]):
            print(f"✗ Greedy: Goal position {goal} is in obstacle")
            return False
        
        return True
    
    def _heuristic(self, a, b):
        """
        Calculate heuristic distance (Euclidean).
        
        Args:
            a (tuple): First position
            b (tuple): Second position
            
        Returns:
            float: Distance estimate
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return np.sqrt(dx**2 + dy**2)
    
    def _calculate_repulsive_potential(self, position):
        """
        Calculate repulsive potential from nearby obstacles.
        Higher values mean closer to obstacles.
        
        Args:
            position (tuple): Current position
            
        Returns:
            float: Repulsive potential
        """
        x, y = position
        repulsive = 0.0
        search_radius = 3  # Check nearby cells
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx, ny = x + dx, y + dy
                
                if not self.map_builder.is_valid_cell(nx, ny):
                    continue
                
                if not self.map_builder.is_cell_free(nx, ny):
                    # Closer obstacles have stronger repulsion
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance > 0:
                        repulsive += 1.0 / distance
        
        return repulsive
    
    def _reconstruct_path(self, came_from, current):
        """
        Reconstruct path from start to goal.
        
        Args:
            came_from (dict): Parent mapping
            current (tuple): Goal position
            
        Returns:
            list: Path from start to goal
        """
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path


# Example usage and testing
if __name__ == "__main__":
    print("Testing Greedy Path Planning Module")
    print("=" * 50)
    
    # Create test map
    from map_builder import MapBuilder
    import time
    
    map_builder = MapBuilder(width=50, height=50)
    
    # Create obstacles
    grid = np.ones((50, 50), dtype=np.uint8)
    grid[15:35, 20:22] = 0  # Vertical wall
    grid[15:17, 10:20] = 0  # Small horizontal barrier
    
    map_builder.update_from_segmentation(grid)
    
    # Create Greedy planner
    planner = GreedyPlanner(map_builder)
    
    # Test both methods
    start = (5, 25)
    goal = (45, 25)
    
    print(f"\nTesting standard greedy search...")
    t0 = time.time()
    path1 = planner.plan(start, goal)
    t1 = time.time()
    
    if path1:
        print(f"✓ Path found with {len(path1)} waypoints")
        print(f"  Planning time: {(t1-t0)*1000:.1f}ms")
    else:
        print("✗ No path found")
    
    print(f"\nTesting potential field approach...")
    t0 = time.time()
    path2 = planner.plan_with_potential_field(start, goal)
    t1 = time.time()
    
    if path2:
        print(f"✓ Path found with {len(path2)} waypoints")
        print(f"  Planning time: {(t1-t0)*1000:.1f}ms")
    else:
        print("✗ No path found")
    
    # Visualize both paths
    if path1 or path2:
        import cv2
        
        if path1:
            vis1 = map_builder.visualize(path=path1, start=start, goal=goal)
            cv2.imshow("Greedy Path", vis1)
        
        if path2:
            vis2 = map_builder.visualize(path=path2, start=start, goal=goal)
            cv2.imshow("Greedy PF Path", vis2)
        
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()