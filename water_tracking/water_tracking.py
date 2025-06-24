import cv2
import mediapipe as mp
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime
import json

@dataclass
class DrinkingEvent:
    """Data class to represent a drinking event"""
    timestamp: float
    volume_ml: float
    sip_count: int
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'volume_ml': self.volume_ml,
            'sip_count': self.sip_count
        }

class WristTracker:
    """Handles wrist position tracking and motion detection"""
    
    def __init__(self, smoothing_window: int = 10, dead_zone: float = 0.01):
        self.smoothing_window = smoothing_window
        self.dead_zone = dead_zone
        self.previous_right_wrist_y: Optional[float] = None
        self.previous_left_wrist_y: Optional[float] = None
        self.right_wrist_history: List[float] = []
        self.left_wrist_history: List[float] = []
    
    def smooth_wrist_position(self, wrist_y_list: List[float]) -> float:
        """Apply smoothing to wrist position data"""
        if len(wrist_y_list) < self.smoothing_window:
            return np.mean(wrist_y_list)
        return np.mean(wrist_y_list[-self.smoothing_window:])
    
    def update_positions(self, right_wrist_y: float, left_wrist_y: float) -> Tuple[float, float]:
        """Update wrist positions and return smoothed values"""
        # Convert to upward-positive coordinate system
        right_y_normalized = 1 - right_wrist_y
        left_y_normalized = 1 - left_wrist_y
        
        # Add to history and smooth
        self.right_wrist_history.append(right_y_normalized)
        self.left_wrist_history.append(left_y_normalized)
        
        smoothed_right = self.smooth_wrist_position(self.right_wrist_history)
        smoothed_left = self.smooth_wrist_position(self.left_wrist_history)
        
        return smoothed_right, smoothed_left
    
    def get_movements(self, current_right: float, current_left: float) -> Tuple[float, float]:
        """Calculate wrist movements with dead zone filtering"""
        if self.previous_right_wrist_y is None or self.previous_left_wrist_y is None:
            self.previous_right_wrist_y = current_right
            self.previous_left_wrist_y = current_left
            return 0.0, 0.0
        
        right_movement = self.previous_right_wrist_y - current_right
        left_movement = self.previous_left_wrist_y - current_left
        
        # Apply dead zone
        if abs(right_movement) < self.dead_zone:
            right_movement = 0.0
        if abs(left_movement) < self.dead_zone:
            left_movement = 0.0
        
        # Update previous positions
        self.previous_right_wrist_y = current_right
        self.previous_left_wrist_y = current_left
        
        return right_movement, left_movement

class CalibrationManager:
    """Handles calibration of drinking motion thresholds"""
    
    def __init__(self, calibration_duration: float = 10.0):
        self.calibration_duration = calibration_duration
        self.is_calibrating = False
        self.calibration_data: List[float] = []
        self.sip_threshold = 0.1
        self.movement_threshold = 0.1
    
    def start_calibration(self):
        """Begin calibration process"""
        print("Calibration started. Please perform a few normal drinking actions for 10 seconds.")
        self.is_calibrating = True
        self.calibration_data = []
        return time.time() + self.calibration_duration
    
    def add_calibration_data(self, right_wrist_y: float):
        """Add data point during calibration"""
        if self.is_calibrating:
            self.calibration_data.append(right_wrist_y)
    
    def finish_calibration(self):
        """Complete calibration and calculate thresholds"""
        self.is_calibrating = False
        
        if not self.calibration_data:
            print("Calibration failed: no data collected.")
            return False
        
        # Calculate thresholds based on calibration data
        mean_position = np.mean(self.calibration_data)
        std_position = np.std(self.calibration_data) if len(self.calibration_data) > 2 else 0.04
        
        self.sip_threshold = mean_position - 0.1
        self.movement_threshold = max(std_position * 6, 0.04)
        
        print(f"Calibration completed successfully!")
        print(f"Sip threshold: {self.sip_threshold:.3f}")
        print(f"Movement threshold: {self.movement_threshold:.3f}")
        return True

class WaterConsumptionTracker:
    """Tracks water consumption based on sip detection"""
    
    def __init__(self, ml_per_sip: float = 15.0, sip_delay: float = 1.0):
        self.ml_per_sip = ml_per_sip  # More realistic than cups conversion
        self.sip_delay = sip_delay
        self.total_sips = 0
        self.last_sip_time = 0.0
        self.drinking_events: List[DrinkingEvent] = []
    
    def detect_sip(self, right_wrist_y: float, right_movement: float,
                   left_wrist_y: float, left_movement: float,
                   sip_threshold: float, movement_threshold: float) -> bool:
        """Detect if a sip occurred based on wrist positions and movements"""
        current_time = time.time()
        
        # Check if enough time has passed since last sip
        if current_time - self.last_sip_time < self.sip_delay:
            return False
        
        # Detect drinking motion (wrist raised above threshold with significant movement)
        right_sip = (right_wrist_y < sip_threshold and 
                    abs(right_movement) > movement_threshold)
        left_sip = (left_wrist_y < sip_threshold and 
                   abs(left_movement) > movement_threshold)
        
        if right_sip or left_sip:
            self.total_sips += 1
            volume_ml = self.total_sips * self.ml_per_sip
            
            # Record drinking event
            event = DrinkingEvent(
                timestamp=current_time,
                volume_ml=volume_ml,
                sip_count=self.total_sips
            )
            self.drinking_events.append(event)
            
            print(f"Sip #{self.total_sips} detected! Total volume: {volume_ml:.1f}ml")
            self.last_sip_time = current_time
            return True
        
        return False
    
    def get_daily_total(self) -> float:
        """Get total water consumption for today"""
        if not self.drinking_events:
            return 0.0
        return self.drinking_events[-1].volume_ml
    
    def save_data(self, filename: str = "water_tracking_data.json"):
        """Save drinking events to JSON file"""
        data = [event.to_dict() for event in self.drinking_events]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {filename}")

class WaterTrackerApp:
    """Main application class that orchestrates all components"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose_detection = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize components
        self.wrist_tracker = WristTracker()
        self.calibration_manager = CalibrationManager()
        self.consumption_tracker = WaterConsumptionTracker()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        
        # App state
        self.tracking_active = False
        self.running = True
        
    def process_frame(self, image: np.ndarray) -> bool:
        """Process a single frame and return whether pose was detected"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = self.pose_detection.process(image_rgb)
        
        if not pose_results.pose_landmarks:
            return False
        
        # Extract wrist positions
        right_wrist = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        # Update wrist tracking
        right_y, left_y = self.wrist_tracker.update_positions(right_wrist.y, left_wrist.y)
        right_movement, left_movement = self.wrist_tracker.get_movements(right_y, left_y)
        
        # Handle calibration
        if self.calibration_manager.is_calibrating:
            self.calibration_manager.add_calibration_data(right_y)
            return True
        
        # Detect sips if tracking is active
        if self.tracking_active:
            sip_detected = self.consumption_tracker.detect_sip(
                right_y, right_movement, left_y, left_movement,
                self.calibration_manager.sip_threshold,
                self.calibration_manager.movement_threshold
            )
            
            # Debug output
            if sip_detected or True:  # Always show debug info
                print(f"Right: Y={right_y:.3f}, Move={right_movement:.4f} | "
                      f"Left: Y={left_y:.3f}, Move={left_movement:.4f}")
        
        return True
    
    def handle_key_input(self, key: int):
        """Handle keyboard input"""
        if key == ord('c'):
            if not self.calibration_manager.is_calibrating:
                calibration_end_time = self.calibration_manager.start_calibration()
                self.calibration_end_time = calibration_end_time
        
        elif key == ord('s'):
            self.tracking_active = not self.tracking_active
            status = "started" if self.tracking_active else "stopped"
            print(f"Tracking {status}.")
        
        elif key == ord('r'):
            # Reset tracking data
            self.consumption_tracker = WaterConsumptionTracker()
            print("Tracking data reset.")
        
        elif key == ord('q') or key == 27:  # 'q' or ESC
            self.running = False
        
        elif key == ord('h'):
            self.show_help()
    
    def show_help(self):
        """Display help information"""
        print("\n" + "="*50)
        print("WATER TRACKER CONTROLS:")
        print("c - Calibrate drinking motion")
        print("s - Start/Stop tracking")
        print("r - Reset tracking data")
        print("h - Show this help")
        print("q/ESC - Quit application")
        print("="*50 + "\n")
    
    def run(self):
        """Main application loop"""
        print("Welcome to the Advanced Water Tracking App!")
        self.show_help()
        
        calibration_end_time = 0
        
        while self.running and self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                continue
            
            # Check if calibration should end
            if (self.calibration_manager.is_calibrating and 
                time.time() > getattr(self, 'calibration_end_time', 0)):
                self.calibration_manager.finish_calibration()
            
            # Process frame
            self.process_frame(image)
            
            # Display status on image
            self.draw_status(image)
            
            # Show image
            cv2.imshow('Water Tracker', image)
            
            # Handle keyboard input
            key = cv2.waitKey(5) & 0xFF
            if key != 255:  # Key was pressed
                self.handle_key_input(key)
        
        self.cleanup()
    
    def draw_status(self, image: np.ndarray):
        """Draw status information on the image"""
        height, width = image.shape[:2]
        
        # Status text
        if self.calibration_manager.is_calibrating:
            status = "CALIBRATING..."
            color = (0, 255, 255)  # Yellow
        elif self.tracking_active:
            status = f"TRACKING - {self.consumption_tracker.get_daily_total():.0f}ml"
            color = (0, 255, 0)  # Green
        else:
            status = "PAUSED"
            color = (0, 0, 255)  # Red
        
        cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, color, 2, cv2.LINE_AA)
        
        # Sip count
        sip_text = f"Sips: {self.consumption_tracker.total_sips}"
        cv2.putText(image, sip_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    def cleanup(self):
        """Clean up resources"""
        # Save data before exit
        if self.consumption_tracker.drinking_events:
            self.consumption_tracker.save_data()
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        total_ml = self.consumption_tracker.get_daily_total()
        print(f"\nSession Summary:")
        print(f"Total sips: {self.consumption_tracker.total_sips}")
        print(f"Total volume: {total_ml:.1f}ml ({total_ml/1000:.2f}L)")
        print("Data saved. Goodbye!")

if __name__ == "__main__":
    app = WaterTrackerApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        app.cleanup()
