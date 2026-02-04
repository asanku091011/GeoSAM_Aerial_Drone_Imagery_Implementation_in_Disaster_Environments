"""
Tello Live Capture - Saves directly to dynamic scene location
Replaces the scene generator with REAL drone footage!
"""

from djitellopy import Tello
import cv2
import time
import os

# IMPORTANT: Save to same location as dynamic_scene_generator.py
OUTPUT_PATH = "data/test_images/current_scene.jpg"

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("="*70)
print("🚁 TELLO LIVE CAPTURE FOR DYNAMIC NAVIGATION")
print("="*70)
print(f"Output: {OUTPUT_PATH}")
print("This replaces the scene generator with REAL drone footage!")
print("="*70)

# Connect to Tello
print("\n[1/3] Connecting to Tello...")
tello = Tello()
tello.connect()

# Check battery
battery = tello.get_battery()
print(f"✓ Connected! Battery: {battery}%")

if battery < 10:
    print("⚠ WARNING: Battery very low! Please charge.")
    exit()

# Start video stream
print("\n[2/3] Starting video stream...")
tello.streamon()
time.sleep(2)  # Wait for stream to stabilize

frame_reader = tello.get_frame_read()
print("✓ Video stream started")

print("\n[3/3] Starting live capture...")
print(f"✓ Saving to: {OUTPUT_PATH}")
print("\nNow you can:")
print("  1. Keep this running")
print("  2. In another terminal, run: python main_dynamic_ultimate.py")
print("  3. The navigation system will use LIVE drone footage!")
print("\nPress Ctrl+C to stop.\n")
print("="*70)

capture_count = 0

try:
    while True:
        # Get current frame
        img = frame_reader.frame
        
        if img is None:
            print("⚠ No frame received, retrying...")
            time.sleep(0.5)
            continue
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Overwrite the same file (same as scene generator!)
        cv2.imwrite(OUTPUT_PATH, img)
        
        capture_count += 1
        print(f"[{capture_count}] Updated: {OUTPUT_PATH} (Battery: {battery}%)")
        
        # Wait 5 seconds before next capture
        time.sleep(5)
        
        # Update battery every 10 captures
        if capture_count % 10 == 0:
            battery = tello.get_battery()
        
except KeyboardInterrupt:
    print("\n\n⚠ Stopping capture...")
    
finally:
    print("\nCleaning up...")
    tello.streamoff()
    tello.end()
    print("✓ Disconnected from Tello")
    print(f"✓ Captured {capture_count} frames total")
    print("\n" + "="*70)
    print("Tello capture stopped.")
    print("="*70)