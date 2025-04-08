import cv2
import mediapipe as mp
import numpy as np
import os
import pygame
import time
from collections import deque

class GesturePuzzle:
    def __init__(self):
        # Initialize MediaPipe with optimized settings for robustness
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,    # Slightly lower to improve detection rate
            min_tracking_confidence=0.7,     # Slightly lower to maintain tracking
            model_complexity=1              # Using more accurate model
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Pygame mixer with high quality settings
        pygame.mixer.pre_init(48000, -16, 2, 512)
        pygame.mixer.init()
        pygame.init()
        pygame.mixer.set_num_channels(16)
        
        # Game state
        self.current_riddle = 0
        self.last_gesture = None
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.3  # Reduced cooldown for better responsiveness
        
        # Gesture recognition improvements
        self.gesture_history = deque(maxlen=3)  # Reduced buffer for faster response
        self.gesture_confidence = 0.0
        self.required_confidence = 0.7  # Lowered threshold for better detection
        
        # Audio file paths
        self.AUDIO_DIR = "audio"
        self.AUDIO_FILES = {
            "final": os.path.join(self.AUDIO_DIR, "final_message.mp3"),
            "riddles": [
                {
                    "question": os.path.join(self.AUDIO_DIR, "riddle_0_question.mp3"),
                    "hint": os.path.join(self.AUDIO_DIR, "riddle_0_hint.mp3"),
                    "success": os.path.join(self.AUDIO_DIR, "riddle_0_success.mp3"),
                    "intro": os.path.join(self.AUDIO_DIR, "riddle_0_intro.mp3"),
                    "answer": "THREE",
                    "gesture_hint": "Show three fingers for firewall layers"
                },
                {
                    "question": os.path.join(self.AUDIO_DIR, "riddle_1_question.mp3"),
                    "hint": os.path.join(self.AUDIO_DIR, "riddle_1_hint.mp3"),
                    "success": os.path.join(self.AUDIO_DIR, "riddle_1_success.mp3"),
                    "intro": os.path.join(self.AUDIO_DIR, "riddle_1_intro.mp3"),
                    "answer": "TWO",
                    "gesture_hint": "Show two fingers for encryption keys"
                },
                {
                    "question": os.path.join(self.AUDIO_DIR, "riddle_2_question.mp3"),
                    "hint": os.path.join(self.AUDIO_DIR, "riddle_2_hint.mp3"),
                    "success": os.path.join(self.AUDIO_DIR, "riddle_2_success.mp3"),
                    "intro": os.path.join(self.AUDIO_DIR, "riddle_2_intro.mp3"),
                    "answer": "FOUR",
                    "gesture_hint": "Show four fingers for authentication factors"
                },
                {
                    "question": os.path.join(self.AUDIO_DIR, "riddle_3_question.mp3"),
                    "hint": os.path.join(self.AUDIO_DIR, "riddle_3_hint.mp3"),
                    "success": os.path.join(self.AUDIO_DIR, "riddle_3_success.mp3"),
                    "intro": os.path.join(self.AUDIO_DIR, "riddle_3_intro.mp3"),
                    "answer": "ONE",
                    "gesture_hint": "Show one finger for local backup"
                },
                {
                    "question": os.path.join(self.AUDIO_DIR, "riddle_4_question.mp3"),
                    "hint": os.path.join(self.AUDIO_DIR, "riddle_4_hint.mp3"),
                    "success": os.path.join(self.AUDIO_DIR, "riddle_4_success.mp3"),
                    "intro": os.path.join(self.AUDIO_DIR, "riddle_4_intro.mp3"),
                    "answer": "FIVE",
                    "gesture_hint": "Show five fingers for cyber kill chain stages"
                }
            ]
        }
        
        # Start with first riddle directly
        self.play_audio(self.AUDIO_FILES["riddles"][0]["intro"])
        time.sleep(0.3)
        self.play_audio(self.AUDIO_FILES["riddles"][0]["question"])

    def play_audio(self, audio_file):
        """Play an audio file directly without caching"""
        try:
            if os.path.exists(audio_file):
                sound = pygame.mixer.Sound(audio_file)
                sound.play()
                # Wait for the audio to finish with a small buffer
                time.sleep(sound.get_length() * 0.95)
                # Clean up the sound object
                del sound
            else:
                print(f"Audio file not found: {audio_file}")
        except Exception as e:
            print(f"Error playing audio: {e}")

    def detect_cross_gesture(self, hand_landmarks1, hand_landmarks2):
        """Enhanced cross gesture detection with improved robustness"""
        if not hand_landmarks1 or not hand_landmarks2:
            return False
            
        try:
            # Get index finger tips and bases
            tip1 = np.array([hand_landmarks1.landmark[8].x, hand_landmarks1.landmark[8].y])
            base1 = np.array([hand_landmarks1.landmark[5].x, hand_landmarks1.landmark[5].y])
            tip2 = np.array([hand_landmarks2.landmark[8].x, hand_landmarks2.landmark[8].y])
            base2 = np.array([hand_landmarks2.landmark[5].x, hand_landmarks2.landmark[5].y])
            
            def is_only_index_extended(landmarks):
                # More robust finger extension check
                index_tip = np.array([landmarks.landmark[8].x, landmarks.landmark[8].y])
                index_pip = np.array([landmarks.landmark[6].x, landmarks.landmark[6].y])
                index_mcp = np.array([landmarks.landmark[5].x, landmarks.landmark[5].y])
                
                # Calculate distances for more accurate detection
                tip_to_pip_dist = np.linalg.norm(index_tip - index_pip)
                pip_to_mcp_dist = np.linalg.norm(index_pip - index_mcp)
                
                # Check other fingers using relative positions
                other_fingers_folded = True
                for tip_id, pip_id in [(12,10), (16,14), (20,18)]:
                    tip = np.array([landmarks.landmark[tip_id].x, landmarks.landmark[tip_id].y])
                    pip = np.array([landmarks.landmark[pip_id].x, landmarks.landmark[pip_id].y])
                    if np.linalg.norm(tip - pip) > 0.1:  # If finger is extended
                        other_fingers_folded = False
                        break
                
                # Improved thumb check
                thumb_tip = np.array([landmarks.landmark[4].x, landmarks.landmark[4].y])
                thumb_ip = np.array([landmarks.landmark[3].x, landmarks.landmark[3].y])
                thumb_folded = np.linalg.norm(thumb_tip - thumb_ip) < 0.1
                
                # More precise index extension check
                index_extended = (tip_to_pip_dist > 0.1) and (pip_to_mcp_dist > 0.05)
                
                return index_extended and other_fingers_folded and thumb_folded
            
            if not (is_only_index_extended(hand_landmarks1) and is_only_index_extended(hand_landmarks2)):
                return False
            
            # Calculate vectors and normalize
            vec1 = tip1 - base1
            vec2 = tip2 - base2
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            
            # Calculate angle with improved accuracy
            angle = np.abs(np.degrees(np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))))
            
            return 60 < angle < 120  # Wider angle range for better detection
            
        except (AttributeError, IndexError) as e:
            return False

    def detect_zero_gesture(self, hand_landmarks):
        """Enhanced zero gesture detection with improved accuracy"""
        if not hand_landmarks:
            return False
            
        try:
            # Get points as numpy arrays for better calculation
            thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
            index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
            
            # Calculate distance using numpy
            distance = np.linalg.norm(thumb_tip - index_tip)
            
            # Check other fingers with improved accuracy
            other_fingers_folded = True
            for tip_id, pip_id in [(12,10), (16,14), (20,18)]:
                tip = np.array([hand_landmarks.landmark[tip_id].x, hand_landmarks.landmark[tip_id].y])
                pip = np.array([hand_landmarks.landmark[pip_id].x, hand_landmarks.landmark[pip_id].y])
                if np.linalg.norm(tip - pip) > 0.08:  # Adjusted threshold
                    other_fingers_folded = False
                    break
            
            return distance < 0.07 and other_fingers_folded  # Slightly relaxed threshold
            
        except (AttributeError, IndexError) as e:
            return False

    def count_fingers(self, hand_landmarks):
        """Enhanced finger counting with improved reliability"""
        if not hand_landmarks:
            return 0
            
        try:
            count = 0
            
            # Improved thumb detection
            thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
            thumb_ip = np.array([hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y])
            thumb_mcp = np.array([hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y])
            
            # Check thumb using angle
            thumb_angle = self._calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
            if thumb_angle > 150:  # Thumb is considered extended if angle is large
                count += 1
            
            # Check other fingers with improved method
            for finger_base, finger_tip in [(5,8), (9,12), (13,16), (17,20)]:  # Index to pinky
                base = np.array([hand_landmarks.landmark[finger_base].x, hand_landmarks.landmark[finger_base].y])
                tip = np.array([hand_landmarks.landmark[finger_tip].x, hand_landmarks.landmark[finger_tip].y])
                mid = np.array([hand_landmarks.landmark[finger_base + 2].x, hand_landmarks.landmark[finger_base + 2].y])
                
                # Calculate angles for better accuracy
                angle = self._calculate_angle(base, mid, tip)
                
                # Consider finger extended if angle is large enough
                if angle > 160:
                    count += 1
            
            return count
            
        except (AttributeError, IndexError) as e:
            return 0

    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        return angle

    def get_gesture_name(self, count):
        """Convert finger count to gesture name"""
        return ['FIST', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'][count]

    def process_gesture(self, results):
        """Enhanced gesture processing with smoothing"""
        current_time = time.time()
        
        if (current_time - self.last_gesture_time) < self.gesture_cooldown:
            return False
            
        # Process two-handed gestures
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            if self.detect_cross_gesture(results.multi_hand_landmarks[0], results.multi_hand_landmarks[1]):
                self.last_gesture_time = current_time
                self.gesture_confidence = 0.0  # Reset confidence for next gesture
                self.play_audio(self.AUDIO_FILES["riddles"][self.current_riddle]["hint"])
                return False
        
        # Process single-handed gestures
        elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Check for zero gesture
            if self.detect_zero_gesture(hand_landmarks):
                self.last_gesture_time = current_time
                self.gesture_confidence = 0.0
                self.play_audio(self.AUDIO_FILES["riddles"][self.current_riddle]["question"])
                return False
            
            # Get current gesture
            finger_count = self.count_fingers(hand_landmarks)
            current_gesture = self.get_gesture_name(finger_count)
            
            # Add to gesture history
            self.gesture_history.append(current_gesture)
            
            # Calculate confidence based on history
            if len(self.gesture_history) >= 3:
                most_common = max(set(self.gesture_history), key=self.gesture_history.count)
                confidence = self.gesture_history.count(most_common) / len(self.gesture_history)
                
                # If we have a stable gesture with high confidence
                if confidence >= self.required_confidence and most_common != self.last_gesture:
                    self.last_gesture = most_common
                    self.last_gesture_time = current_time
                    
                    # Check if it's the correct answer
                    if most_common == self.AUDIO_FILES["riddles"][self.current_riddle]["answer"]:
                        self.gesture_history.clear()
                        self.play_audio(self.AUDIO_FILES["riddles"][self.current_riddle]["success"])
                        time.sleep(0.5)
                        
                        self.current_riddle += 1
                        if self.current_riddle < len(self.AUDIO_FILES["riddles"]):
                            self.play_audio(self.AUDIO_FILES["riddles"][self.current_riddle]["intro"])
                            time.sleep(0.3)
                            self.play_audio(self.AUDIO_FILES["riddles"][self.current_riddle]["question"])
                        else:
                            self.play_audio(self.AUDIO_FILES["final"])
                            return True
        
        return False

    def run(self):
        """Main loop with improved visualization and error handling"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
                
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            # Enhanced visualization with better error handling
            try:
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks with improved styling
                        self.mp_draw.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    if self.process_gesture(results):
                        break
                
                # Enhanced display with better visibility
                if self.current_riddle < len(self.AUDIO_FILES["riddles"]):
                    # Add background rectangle for better text visibility
                    overlay = image.copy()
                    cv2.rectangle(overlay, (5, 5), (800, 40), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
                    
                    # Display riddle info
                    hint_text = f"Riddle {self.current_riddle + 1}/5: {self.AUDIO_FILES['riddles'][self.current_riddle]['gesture_hint']}"
                    cv2.putText(image, hint_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add control reminders with better visibility
                    cv2.rectangle(overlay, (5, image.shape[0]-70), (400, image.shape[0]-10), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
                    
                    cv2.putText(image, "Make 'O' to repeat question", (10, image.shape[0] - 45), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(image, "Cross index fingers for hint", (10, image.shape[0] - 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Cyber Security Riddle Challenge', image)
                
            except Exception as e:
                print(f"Error in visualization: {e}")
                continue
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    puzzle = GesturePuzzle()
    puzzle.run() 