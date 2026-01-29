"""
Data logging system for recording navigation metrics and events.
Logs navigation time, path length, collisions, and replanning frequency.
"""

import csv
import time
import json
from datetime import datetime
import config

class DataLogger:
    """
    Logs navigation performance metrics and events.
    
    This logger records:
    - Navigation execution time
    - Path length (planned and actual)
    - Collision events
    - Replanning frequency
    - Segmentation performance
    - System events and errors
    """
    
    def __init__(self):
        """Initialize the data logger."""
        # Create log directory if needed
        config.create_directories()
        
        # Current session data
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        
        # Metrics storage
        self.navigation_metrics = []
        self.events = []
        
        # Counters
        self.collision_count = 0
        self.replan_count = 0
        self.segmentation_count = 0
        
        # Performance tracking
        self.segmentation_times = []
        self.planning_times = []
        
        print(f"Data Logger initialized (Session: {self.session_id})")
    
    def log_navigation_start(self, start_position, goal_position, algorithm):
        """
        Log the start of a navigation task.
        
        Args:
            start_position (tuple): Start position (x, y)
            goal_position (tuple): Goal position (x, y)
            algorithm (str): Planning algorithm used
        """
        event = {
            'timestamp': time.time(),
            'type': 'navigation_start',
            'start': start_position,
            'goal': goal_position,
            'algorithm': algorithm
        }
        self.events.append(event)
        
        print(f"📊 Logging navigation: {start_position} -> {goal_position} using {algorithm}")
    
    def log_navigation_end(self, success, path_length, execution_time):
        """
        Log the end of a navigation task.
        
        Args:
            success (bool): Whether navigation completed successfully
            path_length (float): Total path length in meters
            execution_time (float): Time taken in seconds
        """
        event = {
            'timestamp': time.time(),
            'type': 'navigation_end',
            'success': success,
            'path_length': path_length,
            'execution_time': execution_time
        }
        self.events.append(event)
        
        # Store metrics for CSV export
        self.navigation_metrics.append({
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'path_length': path_length,
            'execution_time': execution_time,
            'collision_count': self.collision_count,
            'replan_count': self.replan_count
        })
        
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"📊 Navigation ended: {status} ({execution_time:.1f}s, {path_length:.2f}m)")
    
    def log_collision(self, position, obstacle_info=None):
        """
        Log a collision or near-miss event.
        
        Args:
            position (tuple): Position where collision occurred
            obstacle_info (dict): Optional obstacle details
        """
        self.collision_count += 1
        
        event = {
            'timestamp': time.time(),
            'type': 'collision',
            'position': position,
            'obstacle_info': obstacle_info,
            'count': self.collision_count
        }
        self.events.append(event)
        
        print(f"⚠ COLLISION #{self.collision_count} at {position}")
    
    def log_replanning(self, reason, old_path_length, new_path_length):
        """
        Log a replanning event.
        
        Args:
            reason (str): Reason for replanning
            old_path_length (int): Length of old path
            new_path_length (int): Length of new path
        """
        self.replan_count += 1
        
        event = {
            'timestamp': time.time(),
            'type': 'replan',
            'reason': reason,
            'old_path_length': old_path_length,
            'new_path_length': new_path_length,
            'count': self.replan_count
        }
        self.events.append(event)
        
        print(f"🔄 REPLAN #{self.replan_count}: {reason} "
              f"(path: {old_path_length} -> {new_path_length} waypoints)")
    
    def log_segmentation(self, processing_time, safe_percentage):
        """
        Log segmentation performance.
        
        Args:
            processing_time (float): Time taken for segmentation
            safe_percentage (float): Percentage of safe terrain
        """
        self.segmentation_count += 1
        self.segmentation_times.append(processing_time)
        
        event = {
            'timestamp': time.time(),
            'type': 'segmentation',
            'processing_time': processing_time,
            'safe_percentage': safe_percentage,
            'count': self.segmentation_count
        }
        self.events.append(event)
    
    def log_planning(self, algorithm, processing_time, path_length, success):
        """
        Log path planning performance.
        
        Args:
            algorithm (str): Planning algorithm used
            processing_time (float): Time taken for planning
            path_length (int): Number of waypoints in path
            success (bool): Whether planning succeeded
        """
        self.planning_times.append(processing_time)
        
        event = {
            'timestamp': time.time(),
            'type': 'planning',
            'algorithm': algorithm,
            'processing_time': processing_time,
            'path_length': path_length,
            'success': success
        }
        self.events.append(event)
    
    def log_error(self, error_type, message, details=None):
        """
        Log an error or warning.
        
        Args:
            error_type (str): Type of error
            message (str): Error message
            details (dict): Optional error details
        """
        event = {
            'timestamp': time.time(),
            'type': 'error',
            'error_type': error_type,
            'message': message,
            'details': details
        }
        self.events.append(event)
        
        print(f"✗ ERROR ({error_type}): {message}")
    
    def log_custom_event(self, event_type, data):
        """
        Log a custom event.
        
        Args:
            event_type (str): Type of event
            data (dict): Event data
        """
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'data': data
        }
        self.events.append(event)
    
    def save_to_csv(self, filename=None):
        """
        Save navigation metrics to CSV file.
        
        Args:
            filename (str): Output filename (default from config)
            
        Returns:
            bool: True if save successful
        """
        if filename is None:
            filename = config.NAVIGATION_LOG_FILE
        
        if not self.navigation_metrics:
            print("⚠ No navigation metrics to save")
            return False
        
        try:
            # Check if file exists to determine if we need headers
            import os
            file_exists = os.path.exists(filename)
            
            with open(filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'session_id', 'timestamp', 'success', 'path_length',
                    'execution_time', 'collision_count', 'replan_count'
                ])
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Write all metrics
                writer.writerows(self.navigation_metrics)
            
            print(f"✓ Metrics saved to {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save CSV: {str(e)}")
            return False
    
    def save_events(self, filename=None):
        """
        Save all events to a text log file.
        
        Args:
            filename (str): Output filename (default from config)
            
        Returns:
            bool: True if save successful
        """
        if filename is None:
            filename = config.EVENT_LOG_FILE
        
        try:
            with open(filename, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"Session: {self.session_id}\n")
                f.write(f"Started: {datetime.fromtimestamp(self.start_time).isoformat()}\n")
                f.write(f"{'='*70}\n\n")
                
                for event in self.events:
                    timestamp = datetime.fromtimestamp(event['timestamp'])
                    f.write(f"[{timestamp.strftime('%H:%M:%S')}] ")
                    f.write(f"{event['type'].upper()}: ")
                    
                    # Format event-specific details
                    if event['type'] == 'navigation_start':
                        f.write(f"Start navigation from {event['start']} to {event['goal']} ")
                        f.write(f"using {event['algorithm']}\n")
                    
                    elif event['type'] == 'navigation_end':
                        status = "SUCCESS" if event['success'] else "FAILED"
                        f.write(f"{status} - Path: {event['path_length']:.2f}m, ")
                        f.write(f"Time: {event['execution_time']:.1f}s\n")
                    
                    elif event['type'] == 'collision':
                        f.write(f"At position {event['position']} (count: {event['count']})\n")
                    
                    elif event['type'] == 'replan':
                        f.write(f"{event['reason']} - Path changed from ")
                        f.write(f"{event['old_path_length']} to {event['new_path_length']} waypoints\n")
                    
                    elif event['type'] == 'error':
                        f.write(f"{event['error_type']}: {event['message']}\n")
                    
                    else:
                        # Generic format for other events
                        f.write(f"{json.dumps(event, default=str)}\n")
            
            print(f"✓ Events saved to {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save events: {str(e)}")
            return False
    
    def get_summary_statistics(self):
        """
        Generate summary statistics for the session.
        
        Returns:
            dict: Summary statistics
        """
        total_time = time.time() - self.start_time
        
        avg_seg_time = (sum(self.segmentation_times) / len(self.segmentation_times) 
                       if self.segmentation_times else 0)
        avg_plan_time = (sum(self.planning_times) / len(self.planning_times)
                        if self.planning_times else 0)
        
        return {
            'session_id': self.session_id,
            'total_runtime': total_time,
            'navigation_attempts': len(self.navigation_metrics),
            'successful_navigations': sum(1 for m in self.navigation_metrics if m['success']),
            'total_collisions': self.collision_count,
            'total_replans': self.replan_count,
            'segmentation_count': self.segmentation_count,
            'avg_segmentation_time': avg_seg_time,
            'avg_planning_time': avg_plan_time,
            'events_logged': len(self.events)
        }
    
    def print_summary(self):
        """Print a summary of the session statistics."""
        stats = self.get_summary_statistics()
        
        print("\n" + "="*70)
        print("📊 SESSION SUMMARY")
        print("="*70)
        print(f"Session ID: {stats['session_id']}")
        print(f"Runtime: {stats['total_runtime']:.1f}s")
        print(f"\nNavigation:")
        print(f"  Attempts: {stats['navigation_attempts']}")
        print(f"  Successful: {stats['successful_navigations']}")
        print(f"  Success rate: {stats['successful_navigations']/max(stats['navigation_attempts'],1)*100:.1f}%")
        print(f"\nEvents:")
        print(f"  Collisions: {stats['total_collisions']}")
        print(f"  Replans: {stats['total_replans']}")
        print(f"  Total events: {stats['events_logged']}")
        print(f"\nPerformance:")
        print(f"  Segmentations: {stats['segmentation_count']}")
        print(f"  Avg segmentation time: {stats['avg_segmentation_time']*1000:.1f}ms")
        print(f"  Avg planning time: {stats['avg_planning_time']*1000:.1f}ms")
        print("="*70)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Data Logger Module")
    print("=" * 50)
    
    # Create logger
    logger = DataLogger()
    
    # Simulate a navigation session
    print("\nSimulating navigation session...\n")
    
    # Start navigation
    logger.log_navigation_start((10, 10), (90, 90), "A*")
    time.sleep(0.5)
    
    # Log some events
    logger.log_segmentation(0.045, 78.5)
    logger.log_planning("A*", 0.123, 45, True)
    time.sleep(0.3)
    
    # Simulate replanning
    logger.log_replanning("Obstacle detected on path", 45, 52)
    time.sleep(0.2)
    
    # Simulate collision
    logger.log_collision((35, 42))
    time.sleep(0.3)
    
    # End navigation
    logger.log_navigation_end(True, 12.5, 3.2)
    
    # Print summary
    logger.print_summary()
    
    # Save logs
    logger.save_to_csv()
    logger.save_events()
    
    print("\n✓ Test complete!")