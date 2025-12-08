"""
Autonomous Lane Keeping System for slowroads.io

This script integrates:
1. Lane detection (from your existing code)
2. PID control (steering control)
3. Keyboard interface (WASD commands)

Press 'E' to enable autonomous control
Press 'Q' to quit
"""

import cv2
import numpy as np
import time
from pynput import keyboard
import threading

from lane_detector import AdvancedLaneDetector
from pid_controller import LaneKeepingController


class AutonomousDriver:
    """
    Main autonomous driving system that integrates perception and control.
    """
    
    def __init__(self):
        self.detector = AdvancedLaneDetector()
        
        self.controller = LaneKeepingController(
            steering_Kp=2.5,
            steering_Ki=0.08,
            steering_Kd=0.6,
            speed_target=0.7
        )
        
        # System state
        self.running = True
        self.autonomous_mode = False
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
        # Setup keyboard listener for enable/disable
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()
    
    def on_key_press(self, key):
        """Handle keyboard input for enabling/disabling autonomous mode"""
        try:
            if key.char == 'e':
                if not self.autonomous_mode:
                    print("\nAUTONOMOUS MODE ENABLED")
                    self.autonomous_mode = True
                    self.controller.enable_control()
                else:
                    print("\nAUTONOMOUS MODE DISABLED")
                    self.autonomous_mode = False
                    self.controller.disable_control()
            
            elif key.char == 'q':
                print("\nShutting down...")
                self.running = False
                self.controller.disable_control()
        
        except AttributeError:
            # Special keys (like arrows) don't have .char
            pass
    
    def run(self):
        """
        Main control loop:
        1. Capture frame
        2. Detect lanes
        3. Calculate lateral offset
        4. Update PID controller
        5. Apply steering/throttle
        """
        
        print("=" * 60)
        print("AUTONOMOUS LANE KEEPING SYSTEM")
        print("=" * 60)
        print("\nControls:")
        print("  [E] - Toggle autonomous mode ON/OFF")
        print("  [Q] - Quit")
        print("\nWaiting for you to start driving in slowroads.io...")
        print("Press 'E' when ready to enable autonomous control.\n")
        
        time.sleep(5)
        
        while self.running:
            try:
                # STEP 1: PERCEPTION - Detect lanes
                frame = self.detector.capture_screen()
                result, binary, binary_warped = self.detector.process_frame(frame)
                
                # Get lane metrics from detector
                curvature, lateral_offset = self.detector.calculate_curvature(binary_warped)
                
                # Check if lanes are detected
                lanes_detected = (
                    self.detector.left_line.detected and 
                    self.detector.right_line.detected
                )
                
                # STEP 2: CONTROL - Update PID and apply commands
                if self.autonomous_mode and lateral_offset is not None:
                    # PID control based on lateral offset
                    control_info = self.controller.update(
                        lateral_offset=lateral_offset,
                        curvature=curvature,
                        lane_detected=lanes_detected
                    )
                else:
                    # Manual mode or no detection
                    control_info = {
                        'steering_output': 0,
                        'error': lateral_offset if lateral_offset else 0,
                        'P': 0, 'I': 0, 'D': 0,
                        'control_enabled': False
                    }
                    if self.autonomous_mode and lateral_offset is None:
                        print("⚠️  Lane detection lost!")
                
                # ============================================================
                # STEP 3: VISUALIZATION
                # ============================================================
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # Add control information to display
                result = self.add_control_overlay(result, control_info, fps)
                
                # Display windows
                display_height = 400
                aspect_ratio = frame.shape[1] / frame.shape[0]
                display_width = int(display_height * aspect_ratio)
                
                result_resized = cv2.resize(result, (display_width, display_height))
                cv2.imshow('Autonomous Driving', result_resized)
                
                # Optional: Show binary and bird's eye view
                binary_display = cv2.resize(binary * 255, (display_width, display_height))
                cv2.imshow('Binary Threshold', binary_display)
                
                birdeye_display = cv2.resize(binary_warped * 255, (display_width, display_height))
                cv2.imshow("Bird's Eye View", birdeye_display)
                
                # Handle OpenCV window events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️  Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Cleanup
        self.shutdown()
    
    def add_control_overlay(self, frame, control_info, fps):
        """
        Add control information overlay to frame.
        Shows PID terms, steering output, and system status.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Mode indicator
        if self.autonomous_mode:
            mode_text = "AUTO"
            mode_color = (0, 255, 0)
        else:
            mode_text = "MANUAL"
            mode_color = (128, 128, 128)
        
        cv2.putText(frame, f'Mode: {mode_text}', 
                   (frame.shape[1] - 180, 40), font, 0.7, mode_color, 2)
        
        # FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', 
                   (frame.shape[1] - 180, 70), font, 0.6, (0, 255, 255), 2)
        
        # If autonomous mode is enabled, show PID information
        if control_info['control_enabled']:
            y_offset = 150
            
            # Steering output (normalized -1 to 1)
            steering = control_info['steering_output']
            steer_text = f"Steer: {steering:+.3f}"
            steer_color = (0, 255, 0) if abs(steering) < 0.5 else (0, 165, 255)
            cv2.putText(frame, steer_text, 
                       (30, y_offset), font, 0.7, steer_color, 2)
            
            # Error (lateral offset)
            error = control_info['error']
            cv2.putText(frame, f"Error: {error:+.3f}m", 
                       (30, y_offset + 35), font, 0.6, (255, 255, 255), 2)
            
            # PID terms breakdown
            P = control_info['P']
            I = control_info['I']
            D = control_info['D']
            
            cv2.putText(frame, f"P: {P:+.3f}", 
                       (30, y_offset + 70), font, 0.5, (100, 200, 255), 1)
            cv2.putText(frame, f"I: {I:+.3f}", 
                       (30, y_offset + 90), font, 0.5, (100, 200, 255), 1)
            cv2.putText(frame, f"D: {D:+.3f}", 
                       (30, y_offset + 110), font, 0.5, (100, 200, 255), 1)
            
            # Steering indicator bar
            self.draw_steering_bar(frame, steering)
        
        return frame
    
    def draw_steering_bar(self, frame, steering_output):
        """
        Draw a visual bar showing steering direction and magnitude.
        """
        h, w = frame.shape[:2]
        bar_y = h - 50
        bar_width = 300
        bar_height = 20
        bar_x_center = w // 2
        
        # Background bar
        cv2.rectangle(frame, 
                     (bar_x_center - bar_width//2, bar_y),
                     (bar_x_center + bar_width//2, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Center line
        cv2.line(frame,
                (bar_x_center, bar_y),
                (bar_x_center, bar_y + bar_height),
                (255, 255, 255), 2)
        
        # Steering indicator
        steer_x = int(bar_x_center + steering_output * (bar_width // 2))
        indicator_color = (0, 255, 0) if abs(steering_output) < 0.5 else (0, 165, 255)
        
        cv2.rectangle(frame,
                     (bar_x_center, bar_y),
                     (steer_x, bar_y + bar_height),
                     indicator_color, -1)
        
        # Labels
        cv2.putText(frame, 'LEFT', 
                   (bar_x_center - bar_width//2 - 50, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, 'RIGHT', 
                   (bar_x_center + bar_width//2 + 10, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def shutdown(self):
        """Clean shutdown of all systems"""
        print("\n" + "=" * 60)
        print("SHUTTING DOWN")
        print("=" * 60)
        
        self.controller.disable_control()
        cv2.destroyAllWindows()
        self.keyboard_listener.stop()
        
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\nStatistics:")
        print(f"  Frames processed: {self.frame_count}")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Total runtime: {elapsed:.1f}s")
        print("\n✓ Shutdown complete.")


def main():
    """Entry point"""
    driver = AutonomousDriver()
    driver.run()


if __name__ == "__main__":
    main()