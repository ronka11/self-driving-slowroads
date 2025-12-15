import cv2
import numpy as np
import pyautogui
import time
from collections import deque
import scipy
from scipy.ndimage import gaussian_filter1d
import time



class Line:
    """Class to track detected lane line characteristics"""
    def __init__(self, buffer_len=5):
        self.detected = False
        self.recent_fit = deque(maxlen=buffer_len)
        self.best_fit = None
        self.current_fit = None
        self.allx = None
        self.ally = None
        self.fail_count = 0
    
    def update_line(self, fit, allx, ally):
        """Update line with new polynomial fit"""
        if fit is not None:
            if self.best_fit is not None:
                # Check if new fit is reasonable compared to previous
                diff = abs(fit[0] - self.best_fit[0])
                if diff < 0.002:  # Reasonable change threshold
                    self.detected = True
                    self.fail_count = 0
                    self.current_fit = fit
                    self.recent_fit.append(fit)
                    self.best_fit = np.mean(self.recent_fit, axis=0)
                    self.allx = allx
                    self.ally = ally
                else:
                    self.fail_count += 1
                    if self.fail_count > 5:
                        self.detected = False
            else:
                self.detected = True
                self.fail_count = 0
                self.current_fit = fit
                self.recent_fit.append(fit)
                self.best_fit = fit
                self.allx = allx
                self.ally = ally
        else:
            self.fail_count += 1
            if self.fail_count > 10:
                self.detected = False
                self.best_fit = None


class AdvancedLaneDetector:
    def __init__(self):
        self.screen_region = (100, 200, 800, 600)
        
        # Line tracking objects with shorter buffer for responsiveness
        self.left_line = Line(buffer_len=5)
        self.right_line = Line(buffer_len=5)
        
        # Perspective transform matrices
        self.M = None
        self.Minv = None
        
        # Conversion factors (meters per pixel)
        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700
        
    def capture_screen(self):
        """Capture screen region"""
        screenshot = pyautogui.screenshot(region=self.screen_region)
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        return frame
    
    def compute_perspective_transform(self, img_shape):
        """Compute perspective transform matrices for bird's eye view"""
        h, w = img_shape[:2]
        
        # Adjusted source points for better road coverage
        src = np.float32([
            [w * 0.15, h * 0.95],      # Bottom left
            [w * 0.43, h * 0.65],      # Top left
            [w * 0.57, h * 0.65],      # Top right
            [w * 0.85, h * 0.95]       # Bottom right
        ])
        
        # Destination points (rectangle for bird's eye view)
        dst = np.float32([
            [w * 0.25, h],
            [w * 0.25, 0],
            [w * 0.75, 0],
            [w * 0.75, h]
        ])
        
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        
        return self.M, self.Minv
    
    def binarize_frame(self, frame):
        """
        Enhanced binarization specifically tuned for slowroads.io
        Uses histogram equalization for white lanes + HSV for yellow
        """
        h, w = frame.shape[:2]
        
        # Convert to different color spaces
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:,:,2]
        
        # CRITICAL: Histogram equalization for white lane detection
        # This is what the ndrplz repo emphasizes as most important
        l_eq = cv2.equalizeHist(l_channel)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v_clahe = clahe.apply(v_channel)
        
        # Sobel x gradient on equalized L channel
        sobelx_l = cv2.Sobel(l_eq, cv2.CV_64F, 1, 0, ksize=9)
        abs_sobelx = np.absolute(sobelx_l)
        scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-10))
        
        # Gradient thresholds
        sx_binary = np.zeros_like(scaled_sobel)
        sx_binary[(scaled_sobel >= 25) & (scaled_sobel <= 150)] = 1
        
        # White lane detection using equalized L channel (MOST IMPORTANT)
        l_binary = np.zeros_like(l_eq)
        l_binary[(l_eq >= 220) & (l_eq <= 255)] = 1
        
        # Enhanced V channel threshold (white lines)
        v_binary = np.zeros_like(v_clahe)
        v_binary[(v_clahe >= 200) & (v_clahe <= 255)] = 1
        
        # Yellow lane detection using S channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 160) & (s_channel <= 255)] = 1
        
        # Yellow detection in HSV (hue between 15-35 for yellow)
        yellow_binary = np.zeros_like(h_channel)
        yellow_binary[((h_channel >= 15) & (h_channel <= 35)) & (s_channel >= 100)] = 1
        
        # Combine all thresholds with emphasis on equalized channels
        combined_binary = np.zeros_like(sx_binary)
        combined_binary[(l_binary == 1) | (v_binary == 1) | (sx_binary == 1) | 
                       (s_binary == 1) | (yellow_binary == 1)] = 1
        
        # Focus on lower half of image (where lanes are more prominent)
        mask = np.zeros_like(combined_binary)
        mask[int(h*0.4):, :] = 1
        combined_binary = cv2.bitwise_and(combined_binary, mask)
        
        # Morphological operations to clean up
        kernel = np.ones((3,3), np.uint8)
        combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel)
        combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel)
        
        return combined_binary
    
    def find_lane_pixels_histogram(self, binary_warped):
        """
        Find lane pixels using histogram-based sliding window search
        """
        # Take histogram of bottom half
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Smooth the histogram
        histogram = gaussian_filter1d(histogram, sigma=20)
        
        # Find peaks
        midpoint = len(histogram) // 2
        
        # Look for peaks in left and right halves
        left_peak = np.argmax(histogram[:midpoint])
        right_peak = np.argmax(histogram[midpoint:]) + midpoint
        
        # Only proceed if we found significant peaks
        if histogram[left_peak] < 50 or histogram[right_peak] < 50:
            return None, None, None, None
        
        # Sliding window parameters
        nwindows = 9
        margin = 80
        minpix = 30
        
        window_height = binary_warped.shape[0] // nwindows
        
        # Find nonzero pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions
        leftx_current = left_peak
        rightx_current = right_peak
        
        # Lists to receive indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through windows
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Identify pixels in window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # Recenter window
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate indices
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            return None, None, None, None
        
        # Extract positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty
    
    def fit_polynomial(self, binary_warped):
        """
        Fit polynomial to detected lane lines
        """
        leftx, lefty, rightx, righty = self.find_lane_pixels_histogram(binary_warped)
        
        left_fit = None
        right_fit = None
        
        # Fit polynomial only if we have enough points
        if leftx is not None and len(leftx) > 100:
            left_fit = np.polyfit(lefty, leftx, 2)
        
        if rightx is not None and len(rightx) > 100:
            right_fit = np.polyfit(righty, rightx, 2)
        
        # Update line objects
        self.left_line.update_line(left_fit, leftx, lefty)
        self.right_line.update_line(right_fit, rightx, righty)
        
        return left_fit, right_fit
    
    def calculate_curvature(self, binary_warped):
        """
        Calculate radius of curvature and vehicle position
        """
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        
        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit
        
        if left_fit is None or right_fit is None:
            return None, None
        
        # Calculate x values
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        y_eval = np.max(ploty)
        
        # Fit in world space
        left_fit_cr = np.polyfit(ploty*self.ym_per_pix, left_fitx*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*self.ym_per_pix, right_fitx*self.xm_per_pix, 2)
        
        # Calculate radii
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        # Calculate vehicle position
        car_position = binary_warped.shape[1] / 2
        left_lane_bottom = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_lane_bottom = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        lane_center = (left_lane_bottom + right_lane_bottom) / 2
        center_offset = (car_position - lane_center) * self.xm_per_pix
        
        return (left_curverad + right_curverad) / 2, center_offset
    
    def draw_lane(self, frame, binary_warped):
        """
        Draw the detected lane back onto the original frame
        """
        if self.left_line.best_fit is None or self.right_line.best_fit is None:
            return frame
        
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        
        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit
        
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Create polygon points
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw filled lane
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # Draw lane lines
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
        cv2.polylines(color_warp, pts_left, False, (255, 0, 0), 30)
        cv2.polylines(color_warp, pts_right, False, (0, 0, 255), 30)
        
        # Warp back to original
        newwarp = cv2.warpPerspective(color_warp, self.Minv, 
                                     (frame.shape[1], frame.shape[0]))
        
        result = cv2.addWeighted(frame, 1, newwarp, 0.4, 0)
        
        return result
    
    def add_metrics(self, frame, curvature, offset):
        """Add text overlay with lane metrics"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if curvature is not None and curvature < 10000:
            cv2.putText(frame, f'Curve Radius: {int(curvature)}m', 
                       (30, 40), font, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame, 'Curve Radius: Straight', 
                       (30, 40), font, 0.8, (255, 255, 255), 2)
        
        if offset is not None:
            position = 'left' if offset < 0 else 'right'
            cv2.putText(frame, f'Position: {abs(offset):.2f}m {position}', 
                       (30, 75), font, 0.8, (255, 255, 255), 2)
        
        # Detection status
        left_status = 'OK' if self.left_line.detected else 'LOST'
        right_status = 'OK' if self.right_line.detected else 'LOST'
        status_color = (0, 255, 0) if (self.left_line.detected and self.right_line.detected) else (0, 165, 255)
        
        cv2.putText(frame, f'L:{left_status} R:{right_status}', 
                   (30, 110), font, 0.8, status_color, 2)
        
        return frame
    
    def process_frame(self, frame):
        """
        Main processing pipeline
        """
        # Initialize perspective transform
        if self.M is None:
            self.compute_perspective_transform(frame.shape)
        
        # 1. Binarize with histogram equalization
        binary = self.binarize_frame(frame)
        
        # 2. Perspective transform
        binary_warped = cv2.warpPerspective(binary, self.M, 
                                           (frame.shape[1], frame.shape[0]), 
                                           flags=cv2.INTER_LINEAR)
        
        # 3. Detect and fit lanes
        self.fit_polynomial(binary_warped)
        
        # 4. Calculate metrics
        curvature, offset = self.calculate_curvature(binary_warped)
        
        # 5. Draw lane
        result = self.draw_lane(frame, binary_warped)
        
        # 6. Add metrics
        result = self.add_metrics(result, curvature, offset)
        
        return result, binary, binary_warped


def main():
    """
    Main loop for the Lane Detection agent (Perception only).
    1. Captures screen.
    2. Processes frame (Detection, Perspective Transform, Fitting).
    3. Calculates metrics (Curvature and Offset).
    4. Displays annotated result and intermediate views.
    """
    
    detector = AdvancedLaneDetector()
    
    frame_count = 0
    start_time = time.time()
    
    print("Starting Lane Detection Visualization in 3 seconds...")
    time.sleep(3)
    
    while True:
        try:
            # 1. PERCEPTION: Capture and Process Frame
            frame = detector.capture_screen()
            
            # The detector.process_frame handles all image steps (binarization, warp, fit, draw)
            # and returns the final annotated image and intermediate steps.
            result, binary, binary_warped = detector.process_frame(frame)
            
            # The metrics (curvature, offset) are calculated and displayed in add_metrics 
            # within process_frame, so we don't need the explicit control logic here.
            
            # 2. DISPLAY AND TIMING
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Display windows
            display_height = 400
            aspect_ratio = frame.shape[1] / frame.shape[0]
            display_width = int(display_height * aspect_ratio)
            
            # Main result
            result_resized = cv2.resize(result, (display_width, display_height))
            cv2.putText(result_resized, f'FPS: {fps:.1f}', (display_width - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('Lane Detection Result', result_resized)
            
            # Binary threshold
            binary_display = cv2.resize(binary * 255, (display_width, display_height))
            cv2.imshow('Binary Threshold', binary_display)
            
            # Bird's eye
            birdeye_display = cv2.resize(binary_warped * 255, (display_width, display_height))
            cv2.imshow("Bird's Eye View", birdeye_display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.01)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Runtime Error: {e}")
            import traceback
            traceback.print_exc()
            break
            
    # Cleanup on exit
    cv2.destroyAllWindows()
    print(f"\n✓ Processed {frame_count} frames at {fps:.1f} FPS average. Detection stopped.")


# if __name__ == "__main__":
#     main()