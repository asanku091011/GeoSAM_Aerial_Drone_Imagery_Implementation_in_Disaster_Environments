"""
Simple script to view Tello drone camera feed
Press 'q' to quit, 's' to save a snapshot
"""

import cv2
from djitellopy import Tello
import time

print("="*70)
print("🚁 Tello Camera Viewer")
print("="*70)

try:
    # Connect to Tello
    print("\n[1/3] Connecting to Tello...")
    drone = Tello()
    drone.connect()
    
    # Check battery
    battery = drone.get_battery()
    print(f"✓ Connected! Battery: {battery}%")
    
    if battery < 10:
        print("⚠ WARNING: Battery very low! Please charge.")
        exit()
    
    # Start video stream
    print("\n[2/3] Starting video stream...")
    drone.streamon()
    time.sleep(2)  # Wait for stream to stabilize
    
    print("✓ Video stream started")
    
    # Display video
    print("\n[3/3] Displaying camera feed...")
    print("\nControls:")
    print("  Q - Quit")
    print("  S - Save snapshot")
    print("  ESC - Quit")
    print("\n" + "="*70)
    
    frame_count = 0
    
    while True:
        # Get frame
        frame_read = drone.get_frame_read()
        frame = frame_read.frame
        
        if frame is None:
            print("⚠ No frame received, retrying...")
            time.sleep(0.1)
            continue
        
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add info overlay
        frame_count += 1
        cv2.putText(frame, f"Battery: {battery}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'Q' to quit, 'S' to save", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display
        cv2.imshow("Tello Camera Feed", frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("\nQuitting...")
            break
        elif key == ord('s'):  # Save snapshot
            filename = f"tello_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Snapshot saved: {filename}")
    
    # Cleanup
    print("\nCleaning up...")
    cv2.destroyAllWindows()
    drone.streamoff()
    drone.end()
    
    print("✓ Done!")
    
except KeyboardInterrupt:
    print("\n\n⚠ Interrupted by user")
    cv2.destroyAllWindows()
    try:
        drone.streamoff()
        drone.end()
    except:
        pass

except Exception as e:
    print(f"\n✗ Error: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Make sure you're connected to TELLO-XXXXXX WiFi")
    print("2. Make sure drone is powered on (yellow flashing light)")
    print("3. Try turning drone off and on again")
    print("4. Move closer to the drone")
    
    cv2.destroyAllWindows()
    try:
        drone.streamoff()
        drone.end()
    except:
        pass