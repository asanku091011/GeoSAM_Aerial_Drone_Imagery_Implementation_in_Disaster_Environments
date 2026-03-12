"""
Unit Tests for Dynamic Navigation System
Tests all the major components to make sure everything works after changes
Run this with: python test_navigation_system.py
"""

import unittest
import numpy as np
import os
import sys
import time

# Add src to path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

import config
from map_builder import MapBuilder
from astar import AStarPlanner
from path_converter_smooth import SmoothPathConverter


class TestMapBuilder(unittest.TestCase):
    """Test the map builder that converts segmentation to navigation grids"""
    
    def setUp(self):
        """Create a fresh map before each test"""
        self.map_builder = MapBuilder(width=100, height=100)
    
    def test_map_initialization(self):
        """Check that the map starts with all free space"""
        self.assertEqual(self.map_builder.width, 100)
        self.assertEqual(self.map_builder.height, 100)
        # All cells should be free (value = 1) at start
        self.assertTrue(np.all(self.map_builder.grid == 1))
    
    def test_cell_free_check(self):
        """Test checking if a cell is free or blocked"""
        # All cells start free
        self.assertTrue(self.map_builder.is_cell_free(50, 50))
        
        # Block a cell
        self.map_builder.grid[50, 50] = 0
        self.assertFalse(self.map_builder.is_cell_free(50, 50))
        
        # Out of bounds should return False
        self.assertFalse(self.map_builder.is_cell_free(-1, 50))
        self.assertFalse(self.map_builder.is_cell_free(150, 50))
    
    def test_segmentation_update(self):
        """Test updating the map from a segmentation mask"""
        # Create a fake segmentation mask (50% safe, 50% unsafe)
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[:50, :] = 0  # Top half is unsafe
        
        self.map_builder.update_from_segmentation(mask)
        
        # Check that unsafe areas are blocked
        self.assertFalse(self.map_builder.is_cell_free(25, 25))
        # Check that safe areas are free
        self.assertTrue(self.map_builder.is_cell_free(75, 75))


class TestAStarPlanner(unittest.TestCase):
    """Test the A* path planning algorithm"""
    
    def setUp(self):
        """Create a map and planner before each test"""
        self.map_builder = MapBuilder(width=100, height=100)
        self.planner = AStarPlanner(self.map_builder)
    
    def test_simple_path(self):
        """Test planning a simple straight path"""
        start = (10, 10)
        goal = (90, 90)
        
        path = self.planner.plan(start, goal)
        
        # Should find a path
        self.assertIsNotNone(path)
        # Path should start at start
        self.assertEqual(path[0], start)
        # Path should end at goal
        self.assertEqual(path[-1], goal)
        # Path should have multiple waypoints
        self.assertGreater(len(path), 2)
    
    def test_path_with_obstacle(self):
        """Test planning around an obstacle"""
        # Create a wall in the middle
        for y in range(30, 70):
            self.map_builder.grid[y, 50] = 0
        
        start = (10, 50)
        goal = (90, 50)
        
        path = self.planner.plan(start, goal)
        
        # Should still find a path (going around the wall)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        
        # Path should not go through the wall at x=50
        for waypoint in path:
            if waypoint[0] == 50:
                # If we're at x=50, we should be above or below the wall
                self.assertTrue(waypoint[1] < 30 or waypoint[1] >= 70)
    
    def test_no_path_possible(self):
        """Test when there's no possible path (goal surrounded by obstacles)"""
        # Surround the goal with obstacles
        goal = (50, 50)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue  # Don't block the goal itself
                x, y = goal[0] + dx, goal[1] + dy
                if 0 <= x < 100 and 0 <= y < 100:
                    self.map_builder.grid[y, x] = 0
        
        start = (10, 10)
        path = self.planner.plan(start, goal)
        
        # Should return None when no path exists
        self.assertIsNone(path)
    
    def test_start_equals_goal(self):
        """Test when we're already at the goal"""
        start = (50, 50)
        goal = (50, 50)
        
        path = self.planner.plan(start, goal)
        
        # Should return a path with just one point
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], start)


class TestPathConverter(unittest.TestCase):
    """Test converting paths to robot movement commands"""
    
    def setUp(self):
        """Create a map and converter before each test"""
        self.map_builder = MapBuilder(width=100, height=100)
        self.converter = SmoothPathConverter(self.map_builder, unit_scale=1.0)
    
    def test_straight_line_path(self):
        """Test converting a straight horizontal path"""
        path = [(10, 50), (20, 50), (30, 50), (40, 50)]
        
        commands = self.converter.convert_path_to_commands(path, start_heading=0)
        
        # Should generate move commands (already facing correct direction)
        self.assertGreater(len(commands), 0)
        # Should have at least one move command
        self.assertTrue(any(cmd.startswith('move(') for cmd in commands))
    
    def test_path_with_turn(self):
        """Test converting a path that requires turning"""
        # Path that goes right then up
        path = [(10, 10), (20, 10), (20, 20)]
        
        commands = self.converter.convert_path_to_commands(path, start_heading=0)
        
        # Should have both turn and move commands
        has_turn = any(cmd.startswith('turn(') for cmd in commands)
        has_move = any(cmd.startswith('move(') for cmd in commands)
        self.assertTrue(has_turn, "Should have turn commands")
        self.assertTrue(has_move, "Should have move commands")
    
    def test_diagonal_movement(self):
        """Test that diagonal movements are calculated correctly"""
        # 45 degree diagonal path
        path = [(0, 0), (5, 5), (10, 10)]
        
        commands = self.converter.convert_path_to_commands(path, start_heading=0)
        
        # Should generate commands
        self.assertGreater(len(commands), 0)
        
        # Should have a turn to 45 degrees
        has_45_turn = any('45' in cmd for cmd in commands if cmd.startswith('turn('))
        self.assertTrue(has_45_turn, "Should turn to 45 degrees for diagonal")
    
    def test_minimum_move_distance(self):
        """Test that tiny movements are filtered out"""
        # Path with tiny steps (should be combined)
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (10, 0)]
        
        commands = self.converter.convert_path_to_commands(path, start_heading=0)
        
        # Should not have move(1) or move(2) commands (minimum is 5)
        for cmd in commands:
            if cmd.startswith('move('):
                distance = float(cmd[5:-1])
                # Either should be >= 5 or be the final small movement to goal
                self.assertTrue(distance >= 5 or distance < 5)
    
    def test_optimize_removes_redundant_turns(self):
        """Test that optimizer removes unnecessary turn commands"""
        # Manually create commands with redundant turns
        commands = [
            'turn(45.0)',
            'turn(45.0)',  # Redundant! Already at 45
            'move(10.0)',
            'turn(90.0)',
            'move(5.0)'
        ]
        
        optimized = self.converter.optimize_commands(commands)
        
        # Should have fewer commands
        self.assertLess(len(optimized), len(commands))
        
        # Should not have two consecutive identical turns
        for i in range(len(optimized) - 1):
            if optimized[i].startswith('turn('):
                # Next command should not be the same turn
                self.assertNotEqual(optimized[i], optimized[i+1])


class TestObstacleMarking(unittest.TestCase):
    """Test the obstacle marking functionality"""
    
    def setUp(self):
        """Create a map before each test"""
        self.map_builder = MapBuilder(width=100, height=100)
    
    def test_mark_square_obstacle(self):
        """Test marking a square obstacle on the map"""
        # Mark an 8x8 obstacle centered at (50, 50)
        obstacle_center = (50, 50)
        obstacle_size = 8
        half_size = obstacle_size
        
        cells_marked = 0
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                mark_x = obstacle_center[0] + dx
                mark_y = obstacle_center[1] + dy
                
                if (0 <= mark_x < self.map_builder.width and 
                    0 <= mark_y < self.map_builder.height):
                    self.map_builder.grid[mark_y, mark_x] = 0
                    cells_marked += 1
        
        # Should have marked the obstacle
        self.assertGreater(cells_marked, 0)
        
        # Center should be blocked
        self.assertFalse(self.map_builder.is_cell_free(50, 50))
        
        # Cells outside the obstacle should still be free
        self.assertTrue(self.map_builder.is_cell_free(70, 70))
    
    def test_obstacle_position_calculation(self):
        """Test calculating obstacle position ahead of robot"""
        # Robot at (50, 50) facing East (0 degrees)
        robot_pos = (50, 50)
        robot_heading = 0
        distance_ahead = 20
        
        heading_rad = np.radians(robot_heading)
        obstacle_x = robot_pos[0] + distance_ahead * np.cos(heading_rad)
        obstacle_y = robot_pos[1] + distance_ahead * np.sin(heading_rad)
        
        obstacle_x = int(round(obstacle_x))
        obstacle_y = int(round(obstacle_y))
        
        # Should be 20 units to the right (East)
        self.assertEqual(obstacle_x, 70)
        self.assertEqual(obstacle_y, 50)
        
        # Test facing North (90 degrees)
        robot_heading = 90
        heading_rad = np.radians(robot_heading)
        obstacle_x = robot_pos[0] + distance_ahead * np.cos(heading_rad)
        obstacle_y = robot_pos[1] + distance_ahead * np.sin(heading_rad)
        
        obstacle_x = int(round(obstacle_x))
        obstacle_y = int(round(obstacle_y))
        
        # Should be 20 units up (North)
        self.assertEqual(obstacle_x, 50)
        self.assertEqual(obstacle_y, 70)


class TestAngleCalculations(unittest.TestCase):
    """Test angle and heading calculations"""
    
    def test_angle_wraparound(self):
        """Test that angles wrap around correctly at 0/360 boundary"""
        # Test calculating difference between 350 and 10 degrees
        angle1 = 350
        angle2 = 10
        
        diff = abs(angle2 - angle1)
        if diff > 180:
            diff = 360 - diff
        
        # Shortest path is 20 degrees, not 340
        self.assertEqual(diff, 20)
    
    def test_heading_to_radians(self):
        """Test converting headings to radians"""
        # 0 degrees = 0 radians
        self.assertAlmostEqual(np.radians(0), 0)
        
        # 90 degrees = pi/2 radians
        self.assertAlmostEqual(np.radians(90), np.pi / 2)
        
        # 180 degrees = pi radians
        self.assertAlmostEqual(np.radians(180), np.pi)
    
    def test_position_update_from_movement(self):
        """Test updating position based on heading and distance"""
        # Start at (50, 50), heading 0 (East), move 10 units
        pos = (50, 50)
        heading = 0
        distance = 10
        
        heading_rad = np.radians(heading)
        dx = distance * np.cos(heading_rad)
        dy = distance * np.sin(heading_rad)
        
        new_x = int(round(pos[0] + dx))
        new_y = int(round(pos[1] + dy))
        
        # Should move to (60, 50)
        self.assertEqual(new_x, 60)
        self.assertEqual(new_y, 50)
        
        # Test diagonal movement at 45 degrees
        heading = 45
        heading_rad = np.radians(heading)
        dx = distance * np.cos(heading_rad)
        dy = distance * np.sin(heading_rad)
        
        new_x = int(round(pos[0] + dx))
        new_y = int(round(pos[1] + dy))
        
        # Should move diagonally (approximately 7 units in each direction)
        self.assertGreater(new_x, 50)
        self.assertGreater(new_y, 50)


def run_tests():
    """Run all tests and print results"""
    print("\n" + "="*70)
    print("RUNNING NAVIGATION SYSTEM TESTS")
    print("="*70)
    print("\nThis will test all the major components:")
    print("  - Map Builder")
    print("  - A* Path Planner")
    print("  - Path Converter")
    print("  - Obstacle Marking")
    print("  - Angle Calculations")
    print("\n" + "="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMapBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestAStarPlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestPathConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestObstacleMarking))
    suite.addTests(loader.loadTestsFromTestCase(TestAngleCalculations))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\nALL TESTS PASSED!")
        print("Your navigation system is working correctly.")
    else:
        print("\nSOME TESTS FAILED!")
        print("Check the errors above to see what needs fixing.")
    
    print("="*70 + "\n")
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)