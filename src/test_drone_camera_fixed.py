"""
Tello camera viewer with extended timeouts
"""
import cv2
from djitellopy import Tello
import time

print("="*70)
print("🚁 Tello Camera Viewer")
print("="*70)

try:
    print("\n[1/4] Creating Tello object...")
    drone = Tello()
    
    # IMPORTANT: Increase timeout before connecting
    drone.RESPONSE_TIMEOUT = 10  # Increase from default 7 to 10 seconds
    
    print("\n[2/4] Connecting to drone (10 second timeout)...")
    print("Please wait...")
    
    drone.connect()
    print("✓ Connected!")
    
    print("\n[3/4] Getting battery level...")
    battery = drone.get_battery()
    print(f"✓ Battery: {battery}%")
    
    if battery < 10:
        print("⚠ Battery very low!")
        exit()
    
    print("\n[4/4] Starting video stream...")
    print("This takes 10-15 seconds, please wait...")
    
    drone.streamon()
    
    # Give stream plenty of time to start
    for i in range(12, 0, -1):
        print(f"  Waiting... {i} seconds", end='\r')
        time.sleep(1)
    print("\n")
    
    print("✓ Stream should be ready!")
    print("\nControls: Press 'Q' to quit, 'S' to save snapshot")
    print("="*70 + "\n")
    
    frame_count = 0
    got_first_frame = False
    
    while True:
        # Get frame
        frame_read = drone.get_frame_read()
        frame = frame_read.frame
        
        if frame is not None:
            if not got_first_frame:
                print("✓ First frame received!")
                got_first_frame = True
            
            # Convert RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add info
            frame_count += 1
            cv2.putText(frame, f"Battery: {battery}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {frame_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("Tello Camera Feed", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"tello_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Saved: {filename}")
        else:
            print(".", end='', flush=True)
            time.sleep(0.1)
    
    # Cleanup
    print("\n\nShutting down...")
    cv2.destroyAllWindows()
    drone.streamoff()
    drone.end()
    print("✓ Done!")

except KeyboardInterrupt:
    print("\n\nStopped by user")
    cv2.destroyAllWindows()
    drone.streamoff()
    drone.end()

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()