import time
import numpy as np
from pynput.keyboard import Controller, Key
# from pynput.keyboard import Controller as KeyboardController, Key

class PIDController:
    """
    Formula: output = Kp * error + Ki * integral + Kd * derivative
    - Proportional (Kp): Reacts to current error
    - Integral (Ki): Accumulates past errors (eliminates steady-state error)
    - Derivative (Kd): Predicts future error (dampens oscillations)
    """
    def __init__(self, Kp, Ki, Kd, output_limits=(-1,1)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

        self.integral_limit = 1.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update(self, error, current_time=None):
        if current_time is None:
            current_time = time.time()

        if self.prev_time is None:
            dt = 0.0
        else:
            dt = current_time - self.prev_time

        if dt <= 0.0:
            dt = 0.001

        # Proprtional Term
        P = self.Kp * error

        # Integral Term
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        I = self.Ki * self.integral

        # Derivative Term
        self.derivative= (error - self.prev_error) / dt
        D = self.Kd * self.derivative

        output = P + I + D

        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        self.prev_error = error
        self.prev_time = current_time

        return output
    

class LaneKeepingController:
    """
    High-level controller that uses PID to keep the car centered in lane.
    Translates PID output into keyboard commands (WASD).
    """
    def __init__(self, 
                 steering_Kp=2.0,
                 steering_Ki=0.1,
                 steering_Kd=0.5,
                 speed_target=0.6):
        self.steering_pid = PIDController(
            Kp=steering_Kp,
            Ki=steering_Ki,
            Kd=steering_Kd,
            output_limits=(-1,1)
        )

        self.keyboard = Controller()
        # self.keyboard = PIDController()

        self.speed_target = speed_target
        self.current_keys_pressed = set()

        self.max_offset = 2.0
        self.control_enabled = True

    def release_all_keys(self):
        for key in ['w', 'a', 's', 'd']:
            if key in self.current_keys_pressed:
                self.keyboard.release(key)
        self.current_keys_pressed.clear()

    def press_key(self, key):
        if key not in self.current_keys_pressed:
            self.keyboard.press(key)
            self.current_keys_pressed.add(key)

    def release_key(self, key):
        if key in self.current_keys_pressed:
            self.keyboard.release(key)
            self.current_keys_pressed.remove(key)

    def apply_steering(self, steering_output):
        dead_zone = 0.1

        if abs(steering_output) < dead_zone:
            self.release_key('a')
            self.release_key('d')
        elif steering_output < -dead_zone:
            self.release_key('d')
            self.press_key('a')
        else:
            self.release_key('a')
            self.press_key('a')

    def apply_throttle(self, curvature=None):
        self.press_key('w')

    def update(self, lateral_offset, curvature=None, lane_detected=True):
        if not self.control_enabled or not lane_detected:
            self.release_all_keys()
            return {
                'steerting_output': 0,
                'error': lateral_offset,
                'P': 0, 'I': 0, 'D': 0,
                'control_enabled': False
            }
        
        if abs(lateral_offset) > self.max_offset:
            print(f"Large Offset detected, reducing control: {lateral_offset:.2f}m")

        error = lateral_offset
        steering_output = self.steering_pid.update(error)

        self.apply_steering(steering_output)
        self.apply_throttle(curvature)

        return {
            'steering_output': steering_output,
            'error': error,
            'P': self.steering_pid.Kp * error,
            'I': self.steering_pid.Ki * self.steering_pid.integral,
            'D': self.steering_pid.Kd * (error - self.steering_pid.prev_error) / 0.001,
            'control_enabled': True
        }
    
    def enable_control(self):
        self.control_enabled = True
        self.steering_pid.reset()

    def disable_control(self):
        self.control_enabled = False
        self.release_all_keys()
        self.steering_pid.reset()