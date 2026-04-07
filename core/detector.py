import cv2
import numpy as np
import mediapipe as mp

class HandGloveDetector:
    def __init__(self):
        # We use the explicitly supported new MediaPipe Tasks API
        # which is 100x more accurate and natively traces the whole hand contour
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Load the task model file downloaded into the project root
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        
        # Heuristic for glove detection: Skin Color Segmentation.
        # This assumes that if skin is clearly visible, the hand is BARE (No Gloves).
        self.lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        
        self.lower_skin2 = np.array([160, 20, 70], dtype=np.uint8)
        self.upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)

    def is_bare_hand(self, hand_roi):
        """Returns True if the hand ROI looks like bare skin, False if gloves."""
        if hand_roi is None or hand_roi.size == 0:
            return False
            
        # Convert to HSV
        hsv = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
        
        # Create masks for skin tones (handling wrap-around for red hues)
        mask1 = cv2.inRange(hsv, self.lower_skin1, self.upper_skin1)
        mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Calculate the ratio of skin pixels
        skin_pixels = cv2.countNonZero(mask)
        total_pixels = hand_roi.shape[0] * hand_roi.shape[1]
        
        if total_pixels == 0:
            return False
            
        skin_ratio = skin_pixels / total_pixels
        
        # If > 5% of the tight bounding box is skin colored, we consider it a bare hand.
        return skin_ratio > 0.05

    def detect(self, frame):
        """Processes a frame, returns (processed_frame, is_violation)."""
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use MediaPipe format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)
        
        violation_detected = False
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Find the tight bounding box that encapsulates all 21 hand keypoints perfectly!
                x_min_val, y_min_val = w, h
                x_max_val, y_max_val = 0, 0
                
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    if x < x_min_val: x_min_val = x
                    if y < y_min_val: y_min_val = y
                    if x > x_max_val: x_max_val = x
                    if y > y_max_val: y_max_val = y
                
                # Add a smaller padding since this box is highly accurate to the fingers
                pad = 15
                x_min = max(0, x_min_val - pad)
                y_min = max(0, y_min_val - pad)
                x_max = min(w, x_max_val + pad)
                y_max = min(h, y_max_val + pad)
                
                # Extract Hand ROI
                hand_roi = frame[y_min:y_max, x_min:x_max]
                
                # Check for gloves
                bare_hand = self.is_bare_hand(hand_roi)
                
                if bare_hand:
                    violation_detected = True
                    color = (0, 0, 255) # Red for NO gloves
                    label = "Gloves Not Detected"
                else:
                    color = (0, 255, 0) # Green for YES gloves
                    label = "Gloves Detected"
                    
                # Draw Box and Label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
        return frame, violation_detected
