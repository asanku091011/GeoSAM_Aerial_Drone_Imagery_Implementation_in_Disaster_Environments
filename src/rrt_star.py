"""
RRT* (Rapidly-exploring Random Tree Star) path planning algorithm.
RRT* is a sampling-based algorithm that explores the space randomly
and provides asymptotically optimal solutions.
"""

import numpy as np
import config

class RRTStarPlanner:
    """
    RRT* pathfinding algorithm implementation.
    
    RRT* builds a tree by randomly sampling points in the space and
    connecting them to the nearest existing node. The * variant includes
    rewiring to optimize the tree structure and reduce path cost.
    
    Advantages:
    - Works well in high-dimensional spaces
    - Can handle complex obstacle configurations
    - Asymptotically optimal (approaches optimal solution over time)
    """
    
    def __init__(self, map_builder):
        """
        Initialize RRT* planner.
        
        Args:
            map_builder: MapBuilder instance containing the navigation grid
        """
        self.map_builder = map_builder
        self.max_iterations = config.RRT_MAX_ITERATIONS
        self.step_size = config.RRT_STEP_SIZE
        self.goal_sample_rate = config.RRT_GOAL_SAMPLE_RATE
        self.search_radius = config.RRT_SEARCH_RADIUS
        
        print(f"RRT* Planner initialized (max_iter: {self.max_iterations}, step: {self.step_size})")
    
    def plan(self, start, goal):
        """
        Plan a path from start to goal using RRT*.
        
        Args:
            start (tuple): Start position (x, y) in grid coordinates
            goal (tuple): Goal position (x, y) in grid coordinates
            
        Returns:
            list: Path as list of (x, y) tuples, or None if no path exists
        """
        # Validate inputs
        if not self._validate_positions(start, goal):
            return None
        
        # Initialize tree with start node
        self.nodes = [Node(start[0], start[1])]
        self.nodes[0].cost = 0
        
        # Build RRT* tree
        for i in range(self.max_iterations):
            # Sample random point (bias towards goal occasionally)
            if np.random.random() < self.goal_sample_rate:
                random_point = goal
            else:
                random_point = self._sample_random_point()
            
            # Find nearest node in tree
            nearest_idx = self._get_nearest_node_index(random_point)
            nearest_node = self.nodes[nearest_idx]
            
            # Create new node in direction of random point
            new_node = self._steer(nearest_node, random_point)
            
            if new_node is None:
                continue
            
            # Check if path to new node is collision-free
            if not self._is_collision_free(nearest_node, new_node):
                continue
            
            # Find nearby nodes for rewiring
            near_indices = self._find_near_nodes(new_node)
            
            # Choose best parent (lowest cost path to new node)
            new_node = self._choose_parent(new_node, near_indices)
            
            if new_node is None:
                continue
            
            # Add new node to tree
            self.nodes.append(new_node)
            
            # Rewire tree to optimize paths
            self._rewire(new_node, near_indices)
            
            # Check if we can reach goal
            if self._is_near_goal(new_node, goal):
                # Try to connect to goal
                goal_node = Node(goal[0], goal[1])
                if self._is_collision_free(new_node, goal_node):
                    goal_node.parent = len(self.nodes) - 1
                    goal_node.cost = new_node.cost + self._distance(new_node, goal_node)
                    self.nodes.append(goal_node)
                    
                    # Extract path
                    path = self._extract_path(len(self.nodes) - 1)
                    return path
        
        # Max iterations reached without finding path
        print(f"⚠ RRT*: Max iterations ({self.max_iterations}) reached without finding goal")
        
        # Try to return partial path to closest node to goal
        closest_idx = self._get_nearest_node_index(goal)
        if self.nodes[closest_idx].cost < float('inf'):
            path = self._extract_path(closest_idx)
            print(f"  Returning partial path ({len(path)} nodes)")
            return path
        
        return None
    
    def _validate_positions(self, start, goal):
        """Validate start and goal positions."""
        if not self.map_builder.is_valid_cell(start[0], start[1]):
            print(f"✗ RRT*: Start position {start} is out of bounds")
            return False
        
        if not self.map_builder.is_valid_cell(goal[0], goal[1]):
            print(f"✗ RRT*: Goal position {goal} is out of bounds")
            return False
        
        if not self.map_builder.is_cell_free(start[0], start[1]):
            print(f"✗ RRT*: Start position {start} is in obstacle")
            return False
        
        if not self.map_builder.is_cell_free(goal[0], goal[1]):
            print(f"✗ RRT*: Goal position {goal} is in obstacle")
            return False
        
        return True
    
    def _sample_random_point(self):
        """
        Sample a random point in the map space.
        
        Returns:
            tuple: Random (x, y) position
        """
        x = np.random.randint(0, self.map_builder.width)
        y = np.random.randint(0, self.map_builder.height)
        return (x, y)
    
    def _get_nearest_node_index(self, point):
        """
        Find the nearest node in the tree to a given point.
        
        Args:
            point (tuple): Target point (x, y)
            
        Returns:
            int: Index of nearest node
        """
        distances = [self._distance(node, point) for node in self.nodes]
        return np.argmin(distances)
    
    def _steer(self, from_node, to_point):
        """
        Create a new node by moving from from_node towards to_point.
        Movement is limited by step_size.
        
        Args:
            from_node (Node): Starting node
            to_point (tuple): Target point
            
        Returns:
            Node: New node, or None if steering failed
        """
        # Calculate direction vector
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            return None
        
        # Limit step size
        if distance > self.step_size:
            ratio = self.step_size / distance
            dx *= ratio
            dy *= ratio
        
        # Create new node
        new_x = int(from_node.x + dx)
        new_y = int(from_node.y + dy)
        
        # Check if new position is valid
        if not self.map_builder.is_valid_cell(new_x, new_y):
            return None
        
        if not self.map_builder.is_cell_free(new_x, new_y):
            return None
        
        new_node = Node(new_x, new_y)
        new_node.parent = self.nodes.index(from_node)
        new_node.cost = from_node.cost + self._distance(from_node, (new_x, new_y))
        
        return new_node
    
    def _is_collision_free(self, from_node, to_node):
        """
        Check if the path between two nodes is collision-free.
        
        Args:
            from_node (Node): Start node
            to_node (Node): End node
            
        Returns:
            bool: True if path is collision-free
        """
        # Check points along the line
        steps = int(self._distance(from_node, to_node)) + 1
        
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = int(from_node.x + t * (to_node.x - from_node.x))
            y = int(from_node.y + t * (to_node.y - from_node.y))
            
            if not self.map_builder.is_cell_free(x, y):
                return False
        
        return True
    
    def _find_near_nodes(self, new_node):
        """
        Find all nodes within search radius of new_node.
        
        Args:
            new_node (Node): Reference node
            
        Returns:
            list: Indices of nearby nodes
        """
        near_indices = []
        
        for i, node in enumerate(self.nodes):
            if self._distance(node, new_node) <= self.search_radius:
                near_indices.append(i)
        
        return near_indices
    
    def _choose_parent(self, new_node, near_indices):
        """
        Choose the best parent for new_node from nearby nodes.
        Best parent minimizes cost to reach new_node.
        
        Args:
            new_node (Node): New node to connect
            near_indices (list): Indices of nearby nodes
            
        Returns:
            Node: Updated new_node with best parent, or None if no valid parent
        """
        if not near_indices:
            return new_node
        
        costs = []
        
        for i in near_indices:
            near_node = self.nodes[i]
            
            if self._is_collision_free(near_node, new_node):
                cost = near_node.cost + self._distance(near_node, new_node)
                costs.append(cost)
            else:
                costs.append(float('inf'))
        
        min_cost = min(costs)
        
        if min_cost == float('inf'):
            return None
        
        min_idx = near_indices[costs.index(min_cost)]
        new_node.parent = min_idx
        new_node.cost = min_cost
        
        return new_node
    
    def _rewire(self, new_node, near_indices):
        """
        Rewire the tree to reduce costs of nearby nodes.
        
        Args:
            new_node (Node): Newly added node
            near_indices (list): Indices of nearby nodes
        """
        for i in near_indices:
            near_node = self.nodes[i]
            
            # Calculate cost if we reroute through new_node
            new_cost = new_node.cost + self._distance(new_node, near_node)
            
            # If this is better and collision-free, rewire
            if new_cost < near_node.cost and self._is_collision_free(new_node, near_node):
                near_node.parent = len(self.nodes) - 1
                near_node.cost = new_cost
    
    def _is_near_goal(self, node, goal):
        """
        Check if node is close enough to goal to attempt connection.
        
        Args:
            node (Node): Current node
            goal (tuple): Goal position
            
        Returns:
            bool: True if within connection distance
        """
        return self._distance(node, goal) <= self.step_size
    
    def _distance(self, node_or_point1, node_or_point2):
        """
        Calculate Euclidean distance between two nodes or points.
        
        Args:
            node_or_point1: Node or (x, y) tuple
            node_or_point2: Node or (x, y) tuple
            
        Returns:
            float: Euclidean distance
        """
        if isinstance(node_or_point1, Node):
            x1, y1 = node_or_point1.x, node_or_point1.y
        else:
            x1, y1 = node_or_point1
        
        if isinstance(node_or_point2, Node):
            x2, y2 = node_or_point2.x, node_or_point2.y
        else:
            x2, y2 = node_or_point2
        
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _extract_path(self, goal_idx):
        """
        Extract path from start to goal by following parent pointers.
        
        Args:
            goal_idx (int): Index of goal node
            
        Returns:
            list: Path as list of (x, y) tuples
        """
        path = []
        current_idx = goal_idx
        
        while current_idx is not None:
            node = self.nodes[current_idx]
            path.append((node.x, node.y))
            current_idx = node.parent
        
        path.reverse()
        return path


class Node:
    """
    Represents a node in the RRT* tree.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None  # Index of parent node
        self.cost = float('inf')  # Cost from start to this node


# Example usage and testing
if __name__ == "__main__":
    print("Testing RRT* Path Planning Module")
    print("=" * 50)
    
    # Create a test map
    from map_builder import MapBuilder
    import time
    
    map_builder = MapBuilder(width=50, height=50)
    
    # Create complex obstacles
    grid = np.ones((50, 50), dtype=np.uint8)
    grid[15:35, 20:22] = 0  # Vertical wall with gap
    grid[15:17, 20:40] = 0  # Horizontal wall
    
    map_builder.update_from_segmentation(grid)
    
    # Create RRT* planner
    planner = RRTStarPlanner(map_builder)
    
    # Plan path
    start = (5, 25)
    goal = (45, 25)
    
    print(f"\nPlanning path from {start} to {goal}...")
    
    t0 = time.time()
    path = planner.plan(start, goal)
    t1 = time.time()
    
    if path:
        print(f"✓ Path found with {len(path)} waypoints")
        print(f"  Planning time: {(t1-t0)*1000:.1f}ms")
        print(f"  Explored {len(planner.nodes)} nodes")
        
        # Visualize
        import cv2
        vis = map_builder.visualize(path=path, start=start, goal=goal)
        cv2.imshow("RRT* Path", vis)
        
        print("\nPress any key to close window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("✗ No path found")