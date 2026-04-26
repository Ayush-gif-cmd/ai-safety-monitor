import cv2
import numpy as np
import mediapipe as mp
import os
import collections

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class DetectionStabilizer:
    """
    Applies temporal smoothing to predictions (Global).
    Maintains a stable state to prevent UI flicker.
    """
    def __init__(self, history_frames=7, trigger_threshold=0.5):
        self.history_frames = history_frames
        self.trigger_threshold = trigger_threshold
        
        self.head_missing_history = collections.deque(maxlen=history_frames)
        self.mask_missing_history = collections.deque(maxlen=history_frames)
        self.gloves_missing_history = collections.deque(maxlen=history_frames)
        self.people_count_history = collections.deque(maxlen=history_frames)
        self.unprotected_count_history = collections.deque(maxlen=history_frames)

    def _get_stable_boolean(self, history_queue, current_val):
        history_queue.append(current_val)
        if len(history_queue) == 0:
            return current_val
        true_count = sum(1 for v in history_queue if v)
        ratio = true_count / len(history_queue)
        return ratio >= self.trigger_threshold

    def _get_stable_max(self, history_queue, current_val):
        history_queue.append(current_val)
        return max(history_queue) if history_queue else 0

    def stabilize(self, raw_head_missing, raw_mask_missing, raw_gloves_missing, raw_people, raw_unprotected):
        stable_head = self._get_stable_boolean(self.head_missing_history, raw_head_missing)
        stable_mask = self._get_stable_boolean(self.mask_missing_history, raw_mask_missing)
        stable_gloves = self._get_stable_boolean(self.gloves_missing_history, raw_gloves_missing)
        
        stable_people = self._get_stable_max(self.people_count_history, raw_people)
        stable_unprotected = self._get_stable_max(self.unprotected_count_history, raw_unprotected)

        return stable_head, stable_mask, stable_gloves, stable_people, stable_unprotected


class SafetyDetector:
    def __init__(self, yolo_model_path="ppe_model.pt"):
        """
        Initializes the Safety Detector.
        Uses pure YOLO tracking for high accuracy. 
        """
        self.use_yolo_ppe = False
        self.ppe_model = None
        self.base_person_model = None
        self.stabilizer = DetectionStabilizer()
        
        if YOLO_AVAILABLE:
            if os.path.exists(yolo_model_path):
                self.ppe_model = YOLO(yolo_model_path)
                self.use_yolo_ppe = True
                print(f"[Info] YOLOv8 PPE Model loaded. Using Custom YOLO Tracking.")
            else:
                # Load generic YOLO for person tracking, which auto-downloads if missing!
                print("[Info] YOLOv8 PPE model not found. Automatically initializing baseline YOLOv8 counting model.")
                self.base_person_model = YOLO("yolov8n.pt") 
                
                # --- Hands initialization for Fallback inside Person bbox ---
                BaseOptions = mp.tasks.BaseOptions
                HandLandmarker = mp.tasks.vision.HandLandmarker
                HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode
                
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
                    running_mode=VisionRunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.4,
                    min_hand_presence_confidence=0.4
                )
                self.hand_landmarker = HandLandmarker.create_from_options(options)

                # Skin color matrices
                self.lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
                self.upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
                self.lower_skin2 = np.array([160, 20, 70], dtype=np.uint8)
                self.upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        else:
            print("[Warning] Ultralytics module missing. Please check installations.")

    def is_bare_skin(self, roi, threshold=0.05):
        if roi is None or roi.size == 0: return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_skin1, self.upper_skin1)
        mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)
        hair_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
        combined_mask = cv2.bitwise_or(mask, hair_mask)

        skin_pixels = cv2.countNonZero(combined_mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0: return False
        
        return (skin_pixels / total_pixels) > threshold

    def detect(self, frame):
        """Processes a frame using the active active logic."""
        if not YOLO_AVAILABLE:
            return frame, [], 0, 0
            
        if self.use_yolo_ppe:
            frame, r_head, r_mask, r_gloves, r_people, r_unp = self.detect_full_yolo(frame)
        else:
            frame, r_head, r_mask, r_gloves, r_people, r_unp = self.detect_hybrid_yolo_mediapipe(frame)

        stable_h, stable_m, stable_g, stable_p, stable_u = self.stabilizer.stabilize(r_head, r_mask, r_gloves, r_people, r_unp)
        
        violations = set()
        if stable_h: violations.add("Head Protection Missing")
        if stable_m: violations.add("Mask Missing")
        if stable_g: violations.add("Gloves Missing")

        return frame, list(violations), stable_p, stable_u

    def detect_full_yolo(self, frame):
        """Pure PPE Model Tracking"""
        results = self.ppe_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.45)
        raw_head, raw_mask, raw_gloves, unprotected_count = False, False, False, 0
        
        unique_people = set()
        unique_unprotected = set()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                
                label_name = self.ppe_model.names[cls].lower()
                color = (0, 255, 0)
                
                if "person" in label_name:
                    unique_people.add(track_id)
                
                person_unprotected = False
                
                if any(x in label_name for x in ["no_helmet", "head", "bare_head", "no_cap"]):
                    raw_head = True
                    person_unprotected = True
                    color = (0, 0, 255)
                    label_name = "No Helmet"
                elif any(x in label_name for x in ["no_mask", "bare_face", "face_exposed"]):
                    raw_mask = True
                    person_unprotected = True
                    color = (0, 0, 255)
                    label_name = "No Mask"
                elif any(x in label_name for x in ["no_gloves", "hand", "bare_hand"]):
                    raw_gloves = True
                    person_unprotected = True
                    color = (0, 0, 255)
                    label_name = "No Gloves"
                elif "helmet" in label_name or "cap" in label_name:
                    label_name = "Helmet/Cap ✅"
                elif "mask" in label_name:
                    label_name = "Mask ✅"
                elif "glove" in label_name:
                    label_name = "Gloves ✅"
                
                if person_unprotected and track_id != -1:
                    unique_unprotected.add(track_id)
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label_name}{' ID:'+str(track_id) if track_id != -1 else ''}"
                cv2.putText(frame, text, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, raw_head, raw_mask, raw_gloves, len(unique_people), len(unique_unprotected)

    def detect_hybrid_yolo_mediapipe(self, frame):
        """Base Person Tracking (YOLOv8n) + Internal Bbox Fallback"""
        results = self.base_person_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.50)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        raw_head = False
        raw_mask = False
        raw_gloves = False
        
        unique_people = set()
        unique_unprotected = set()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Ensure it's a 'person' (COCO class 0)
                if int(box.cls[0]) != 0: continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                
                unique_people.add(track_id)
                person_unprotected = False
                color = (0, 255, 0)
                
                # Extract localized Person Region of Interest
                ph = y2 - y1
                pw = x2 - x1
                person_roi = frame[y1:y2, x1:x2]
                
                # --- Head Checking within Person Box ---
                head_roi_y_min = 0
                head_roi_y_max = int(ph * 0.25) # Top 25% of body box
                head_roi = person_roi[head_roi_y_min:head_roi_y_max, :]
                
                head_abs_y_min = y1
                head_abs_y_max = y1 + head_roi_y_max
                
                person_bare_head = self.is_bare_skin(head_roi, threshold=0.10)
                if person_bare_head:
                    raw_head = True
                    person_unprotected = True
                    head_color = (0, 0, 255)
                    head_label = "No Helmet"
                else:
                    head_color = (0, 255, 0)
                    head_label = "Helmet ✅"
                
                # Draw Head Tight Box
                cv2.rectangle(frame, (x1, head_abs_y_min), (x2, head_abs_y_max), head_color, 2)
                cv2.putText(frame, head_label, (x1, max(15, head_abs_y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, head_color, 2)
                    
                # --- Mask Checking within Person Box ---
                mask_roi_y_min = int(ph * 0.25) # the next 15% for the lower face
                mask_roi_y_max = int(ph * 0.40)
                mask_roi = person_roi[mask_roi_y_min:mask_roi_y_max, :]
                
                mask_abs_y_min = y1 + mask_roi_y_min
                mask_abs_y_max = y1 + mask_roi_y_max
                
                person_bare_face = self.is_bare_skin(mask_roi, threshold=0.10)
                if person_bare_face:
                    raw_mask = True
                    person_unprotected = True
                    mask_color = (0, 0, 255)
                    mask_label = "No Mask"
                else:
                    mask_color = (0, 255, 0)
                    mask_label = "Mask ✅"
                
                # Draw Mask Tight Box
                cv2.rectangle(frame, (x1, mask_abs_y_min), (x2, mask_abs_y_max), mask_color, 2)
                cv2.putText(frame, mask_label, (x1, max(15, mask_abs_y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color, 2)
                
                # --- Hand Checking within Person Box ---
                # To keep performance insanely fast, we only pass the person ROI to MediaPipe, masking out everything else.
                blank_frame = np.zeros_like(rgb_frame)
                blank_frame[y1:y2, x1:x2] = rgb_frame[y1:y2, x1:x2]
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=blank_frame)
                hand_result = self.hand_landmarker.detect(mp_image)
                
                hands_protected = True
                if hand_result.hand_landmarks:
                    for land in hand_result.hand_landmarks:
                        hx_min_v, hy_min_v = w, h
                        hx_max_v, hy_max_v = 0, 0
                        
                        for lm in land:
                            lx, ly = int(lm.x * w), int(lm.y * h)
                            hx_min_v, hy_min_v = min(hx_min_v, lx), min(hy_min_v, ly)
                            hx_max_v, hy_max_v = max(hx_max_v, lx), max(hy_max_v, ly)
                            
                        # Crop tight hand bounding box
                        hx_min, hy_min = max(0, hx_min_v-10), max(0, hy_min_v-10)
                        hx_max, hy_max = min(w, hx_max_v+10), min(h, hy_max_v+10)
                        
                        hand_roi = frame[hy_min:hy_max, hx_min:hx_max]
                        bare = self.is_bare_skin(hand_roi, threshold=0.10)
                        
                        hand_color = (0, 0, 255) if bare else (0, 255, 0)
                        hand_label = "No Gloves" if bare else "Gloves ✅"
                        
                        if bare: 
                            hands_protected = False
                            raw_gloves = True
                            person_unprotected = True
                        
                        # Draw tight hand box
                        cv2.rectangle(frame, (hx_min, hy_min), (hx_max, hy_max), hand_color, 2)
                        cv2.putText(frame, hand_label, (hx_min, max(15, hy_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)
                
                if person_unprotected:
                    unique_unprotected.add(track_id)
                    color = (0, 0, 255) # Turn entire person box red if unprotected
                
                # Base Person Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1) # Thinner line for person box
                id_text = f"P-ID:{track_id}" if track_id != -1 else "Person"
                cv2.putText(frame, id_text, (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        total_people = len(unique_people)
        total_unprotected = len(unique_unprotected)
        
        return frame, raw_head, raw_mask, raw_gloves, total_people, total_unprotected
