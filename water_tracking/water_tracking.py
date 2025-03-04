import cv2
import mediapipe as mp
import time
import numpy as np
#import streamlit as st

mp_pose = mp.solutions.pose
pose_detection = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

sip_threshold = 0.1
total_sips = 0
previous_right_wrist_y = None
previous_left_wrist_y = None
smoothing_window = 10
calibrated_sip_threshold = sip_threshold
calibrated_wrist_movement_threshold = 0.1
calibration_mode = False
water_data = []
calibration_data_right = []
calibration_data_left = []
last_sip_time = 0
sip_delay = 1 #minimum delay between sips in seconds.
tracking_active = False  # Added tracking state

def estimate_cups(sips):
    return sips / 13

def smooth_wrist_y(wrist_y_list):
    if len(wrist_y_list) < smoothing_window:
        return np.mean(wrist_y_list)
    else:
        return np.mean(wrist_y_list[-smoothing_window:])

def calibrate():
    global calibrated_sip_threshold, calibrated_wrist_movement_threshold, calibration_mode, calibration_data_right, calibration_data_left
    print("Calibration started. Please perform a few normal drinking actions.")
    calibration_data_right = []
    calibration_data_left = []
    calibration_mode = True
    calibration_time = time.time() + 10
    while time.time() < calibration_time:
        success, image = cap.read()
        if not success:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = pose_detection.process(image_rgb)
        if pose_results.pose_landmarks:
            right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            calibration_data_right.append(1 - right_wrist.y)

    if calibration_data_right:
        calibrated_sip_threshold = np.mean(calibration_data_right) - 0.1
        if(len(calibration_data_right) > 2):
            calibrated_wrist_movement_threshold = np.std(calibration_data_right) * 6
            calibrated_wrist_movement_threshold = max(calibrated_wrist_movement_threshold, 0.04)
        print("Calibration finished.")
        print(f"Calibrated sip threshold: {calibrated_sip_threshold:.3f}")
        print(f"Calibrated wrist movement threshold: {calibrated_wrist_movement_threshold:.3f}")
    else:
        print("Calibration failed, no data collected.")
    calibration_mode = False

#if st.button("Start/Stop Tracking"):
#    tracking_active = not tracking_active
#    if tracking_active:
#        st.write("Tracking started.")
#    else:
#        st.write("Tracking stopped.")

#if st.button("Calibrate"):
#    calibrate()

#placeholder = st.empty() #create placeholder for dynamic text.

print("Welcome to the Water Tracking App!")
print("Press 'c' to calibrate, 's' to start/stop tracking.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose_detection.process(image_rgb)

    if pose_results.pose_landmarks:
        right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        right_wrist_y = smooth_wrist_y([1 - right_wrist.y])
        left_wrist_y = smooth_wrist_y([1 - left_wrist.y])

        if previous_right_wrist_y is not None and previous_left_wrist_y is not None:
            right_wrist_movement = previous_right_wrist_y - right_wrist_y
            left_wrist_movement = previous_left_wrist_y - left_wrist_y

            dead_zone = 0.01
            if abs(right_wrist_movement) < dead_zone:
                right_wrist_movement = 0
            if abs(left_wrist_movement) < dead_zone:
                left_wrist_movement = 0

            if tracking_active and not calibration_mode:
                current_time = time.time()
                # Use AND condition and add a delay between sips
                if (current_time - last_sip_time > sip_delay) and ((right_wrist_y < calibrated_sip_threshold and abs(right_wrist_movement) > calibrated_wrist_movement_threshold) or (left_wrist_y < calibrated_sip_threshold and abs(left_wrist_movement) > calibrated_wrist_movement_threshold)):

                    total_sips += 1
                    print(f"Sip detected! Total sips: {total_sips}")
                    cups = estimate_cups(total_sips)
                    print(f"Estimated cups of water: {cups:.2f}")
                    water_data.append((current_time, cups))
                    last_sip_time = current_time

                print(f"Right movement: {right_wrist_movement:.4f}, Right Y: {right_wrist_y:.4f}, Left movement: {left_wrist_movement:.4f}, Left Y: {left_wrist_y:.4f}")

        previous_right_wrist_y = right_wrist_y
        previous_left_wrist_y = left_wrist_y

    cv2.imshow('MediaPipe Detection', image)
    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):
        calibrate()
    elif key == ord('s'):
            tracking_active = not tracking_active
            if tracking_active:
                print("Tracking started.")
            else:
                print("Tracking stopped.")

cap.release()
cv2.destroyAllWindows()

print("\nWater Consumption Data:")
for timestamp, cups in water_data:
    print(f"Time: {time.ctime(timestamp)}, Cups: {cups:.2f}")