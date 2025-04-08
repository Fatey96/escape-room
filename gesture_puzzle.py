import cv2
import mediapipe as mp
import numpy as np
import os
from openai import OpenAI
import pygame
from dotenv import load_dotenv
import time
from pathlib import Path
import tempfile

# Load environment variables
load_dotenv()

# Constants
# Create an 'audio' directory in the current working directory
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Initialize Pygame mixer at the module level
try:
    pygame.mixer.pre_init(44100, -16, 2, 2048)
    pygame.mixer.init()
    pygame.init()
except Exception as e:
    print(f"Warning: Could not initialize pygame audio: {e}")

RIDDLES = [
    {
        "text": "Welcome, brave seeker of digital wisdom! Before you lies the path of cyber enlightenment. Five trials await, each testing your understanding of the sacred arts. Are you prepared to begin this journey?",
        "question": "In the realm of cyber defense, I stand as the first line,\nBlocking unwanted visitors, like sentries divine.\nHow many layers of protection do I typically provide?\nShow with your fingers, let wisdom be your guide!",
        "hint": "Think of the three guardians: One who watches the packets flow, one who guards the gates, and one who knows the applications.",
        "answer": "THREE",  # Representing firewall's three main layers
        "success": "Indeed! You have identified the three sacred layers - packet filtering, circuit gateway, and application gateway. Your understanding of the defensive arts grows.",
        "gesture_hint": "Show three fingers to represent the three layers of firewall protection."
    },
    {
        "text": "Your wisdom serves you well! The second trial beckons.",
        "question": "In the dance of encryption, partners we need,\nOne to lock secrets, one to succeed.\nHow many keys does asymmetric encryption employ?\nShow the count, let your wisdom deploy!",
        "hint": "Consider the duality of security - what is locked by one can only be unlocked by its other half.",
        "answer": "TWO",  # Public and private keys
        "success": "Yes! The public and private keys - like the sun and moon, they work in harmony to guard our secrets. Your cyber wisdom grows stronger!",
        "gesture_hint": "Show two fingers for the two keys in asymmetric encryption."
    },
    {
        "text": "Ah, you show promise! Now face the third challenge.",
        "question": "In the realm of passwords, strength we seek,\nFactors of authentication, make hackers weak.\nHow many factors make authentication strong?\nShow the number, prove you belong!",
        "hint": "Think of what you know, what you have, what you are, and where you stand.",
        "answer": "FOUR",  # Something you know, have, are, and location
        "success": "Excellent! Knowledge, possession, inherence, and location - the four pillars of authentication. You truly grasp the foundations of security!",
        "gesture_hint": "Show four fingers for the four authentication factors."
    },
    {
        "text": "Your knowledge runs deep. But can you solve this riddle?",
        "question": "Single point of failure, we must avoid,\nIn backup strategies, wisdom employed.\nThe minimum backup copies wise ones keep,\nShow this number, let not data sleep!",
        "hint": "In the ancient rule of 3-2-1, we speak of the one copy that stays closest to home.",
        "answer": "ONE",  # 3-2-1 rule, but showing 1 for the local copy
        "success": "Indeed! The local copy - the first guardian in the sacred 3-2-1 rule of backup! Your wisdom in data protection shines bright!",
        "gesture_hint": "Show one finger for the local backup copy."
    },
    {
        "text": "The final test stands before you. Prove your worth!",
        "question": "In cyber's sacred pentagon of death,\nStages of attack make defenders hold their breath.\nShow the phases of this killing chain,\nProve your worth through this final strain!",
        "hint": "Count the steps of the attacker's path: they must first see, then arm themselves, deliver their weapon, breach the walls, and establish their presence.",
        "answer": "FIVE",  # Five stages of cyber kill chain
        "success": "EXTRAORDINARY! You have mastered the five stages: reconnaissance, weaponization, delivery, exploitation, and installation! You are truly worthy!",
        "gesture_hint": "Show five fingers for the five stages of the cyber kill chain."
    }
]

FINAL_MESSAGE = "You have proven yourself a true master of cyber wisdom! The ancient firewall parts, and reveals your reward - the sacred PIN of passage: 32145. May you use this knowledge wisely, for with great power comes great responsibility. You may now pass into the realm of the cyber elite!"

class GesturePuzzle:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Allow detection of two hands for cross gesture
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        self.client = OpenAI(api_key=api_key)
        
        # Game state
        self.current_riddle = 0
        self.last_gesture = None
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0  # Cooldown in seconds
        
        # Pre-generate all audio files
        self.generate_all_audio_files()
        
        # Play welcome message
        self.play_audio_file("welcome.mp3")
        time.sleep(2)
        self.play_audio_file(f"riddle_{self.current_riddle}_question.mp3")

    def generate_all_audio_files(self):
        """Pre-generate all audio files"""
        print("Generating all audio files...")
        
        # Generate welcome message with extra Gandalf-like flourish
        welcome_text = (
            "Ah... Welcome, brave seeker of digital wisdom!  "
            "Before you lies the path of cyber enlightenment. *pause* "
            "Five trials await, each testing your understanding of the sacred arts. *pause* "
            "Are you... prepared to begin this journey?"
        )
        self.generate_audio_file(welcome_text, "welcome.mp3")
        
        # Generate audio for each riddle
        for i, riddle in enumerate(RIDDLES):
            # Add Gandalf-like dramatic pauses and emphasis to each component
            intro_text = riddle["text"].replace("!", "... !")
            self.generate_audio_file(intro_text, f"riddle_{i}_intro.mp3")
            
            question_text = riddle["question"].replace("\n", "... *pause* ")
            self.generate_audio_file(question_text, f"riddle_{i}_question.mp3")
            
            hint_text = f"Mmm... {riddle['hint']}"  # Add thoughtful Gandalf-like "Mmm"
            self.generate_audio_file(hint_text, f"riddle_{i}_hint.mp3")
            
            success_text = f"*pause* {riddle['success']}"
            self.generate_audio_file(success_text, f"riddle_{i}_success.mp3")
        
        # Generate final message with extra gravitas
        final_text = (
            "You have proven yourself... a true master of cyber wisdom! *pause* "
            "The ancient firewall parts... and reveals your reward... "
            "the sacred PIN of passage: *pause* 3... 2... 1... 4... 5. *pause* "
            "May you use this knowledge wisely... for with great power... "
            "comes great responsibility. *pause* "
            "You may now pass... into the realm of the cyber elite!"
        )
        self.generate_audio_file(final_text, "final_message.mp3")
        print("Audio generation complete!")

    def generate_audio_file(self, text, filename):
        """Generate a single audio file"""
        try:
            print(f"Generating {filename}...")
            
            # Add dramatic pauses and emphasis for Gandalf-like speech
            text = text.replace("!", "... !")  # Add dramatic pauses before exclamations
            text = text.replace(".", "... .")  # Add dramatic pauses between sentences
            text = text.replace("\n", "... ")  # Add dramatic pauses at line breaks
            
            response = self.client.audio.speech.create(
                model="tts-1-hd",  # Using HD model for better quality
                voice="onyx",  # Deep, authoritative voice
                speed=0.85,  # Slightly slower for more gravitas
                input=text
            )
            
            filepath = os.path.join(AUDIO_DIR, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Generated {filename}")
        except Exception as e:
            print(f"Error generating {filename}: {e}")

    def play_audio_file(self, filename):
        """Play a specific audio file"""
        try:
            filepath = os.path.join(AUDIO_DIR, filename)
            if os.path.exists(filepath):
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                
                pygame.mixer.music.stop()
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            else:
                print(f"Audio file not found: {filepath}")
        except Exception as e:
            print(f"Error playing audio: {e}")

    def detect_cross_gesture(self, hand_landmarks1, hand_landmarks2):
        """Detect if two hands are making a cross with index fingers"""
        if not hand_landmarks1 or not hand_landmarks2:
            return False
            
        # Get index finger tips
        tip1 = hand_landmarks1.landmark[8]
        tip2 = hand_landmarks2.landmark[8]
        
        # Check if only index fingers are extended
        def is_only_index_extended(landmarks):
            index_tip = landmarks.landmark[8].y
            index_pip = landmarks.landmark[6].y
            other_fingers = [landmarks.landmark[i].y for i in [12, 16, 20]]
            return index_tip < index_pip and all(tip > index_pip for tip in other_fingers)
        
        # Check if both hands have only index fingers extended and they cross
        if (is_only_index_extended(hand_landmarks1) and 
            is_only_index_extended(hand_landmarks2)):
            # Simple crossing check
            return abs(tip1.x - tip2.x) < 0.1 and abs(tip1.y - tip2.y) < 0.1
            
        return False

    def detect_zero_gesture(self, hand_landmarks):
        """Detect if hand is making a zero gesture (thumb and index touching)"""
        if not hand_landmarks:
            return False
            
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        # Check if thumb and index are touching
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        return distance < 0.05

    def process_gesture(self, results):
        """Process detected hand gestures"""
        current_time = time.time()
        
        if (current_time - self.last_gesture_time) < self.gesture_cooldown:
            return False
            
        # Check for two hands (cross gesture)
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            if self.detect_cross_gesture(results.multi_hand_landmarks[0], results.multi_hand_landmarks[1]):
                self.last_gesture_time = current_time
                # Play hint for current riddle
                self.play_audio_file(f"riddle_{self.current_riddle}_hint.mp3")
                return False
        
        # Check for single hand gestures
        elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Check for zero gesture (repeat question)
            if self.detect_zero_gesture(hand_landmarks):
                self.last_gesture_time = current_time
                self.play_audio_file(f"riddle_{self.current_riddle}_question.mp3")
                return False
            
            # Check for answer gesture
            finger_count = self.count_fingers(hand_landmarks)
            gesture = self.get_gesture_name(finger_count)
            
            if gesture != self.last_gesture:
                self.last_gesture = gesture
                self.last_gesture_time = current_time
                
                # Check if gesture matches current riddle's answer
                if gesture == RIDDLES[self.current_riddle]["answer"]:
                    # Play success message
                    self.play_audio_file(f"riddle_{self.current_riddle}_success.mp3")
                    time.sleep(2)
                    
                    self.current_riddle += 1
                    if self.current_riddle < len(RIDDLES):
                        # Play next riddle introduction and question
                        self.play_audio_file(f"riddle_{self.current_riddle}_intro.mp3")
                        time.sleep(1)
                        self.play_audio_file(f"riddle_{self.current_riddle}_question.mp3")
                    else:
                        # Complete the game
                        self.play_audio_file("final_message.mp3")
                        return True
        
        return False

    def count_fingers(self, hand_landmarks):
        """Count fingers and return the gesture number"""
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tip
        thumb_tip = 4
        count = 0
        
        # Check thumb
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
            count += 1
            
        # Check other fingers
        for tip in finger_tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                count += 1
                
        return count

    def get_gesture_name(self, count):
        """Convert finger count to gesture name"""
        return ['FIST', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'][count]

    def run(self):
        """Main loop for gesture detection"""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
                
            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Process the gesture
                    if self.process_gesture(results):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
            
            # Display current riddle number and gesture hint
            if self.current_riddle < len(RIDDLES):
                hint_text = f"Riddle {self.current_riddle + 1}/5: {RIDDLES[self.current_riddle]['gesture_hint']}"
                cv2.putText(image, hint_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the image
            cv2.imshow('Cyber Security Riddle Challenge', image)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    puzzle = GesturePuzzle()
    puzzle.run() 