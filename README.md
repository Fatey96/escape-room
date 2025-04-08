# Simple Gesture Puzzle with Custom Audio

A simplified version of the cyber security riddle challenge that uses pre-recorded audio files and hand gestures.

## Setup

1. Create an `audio` directory in the same folder as the script
2. Add your pre-recorded audio files with these names:
   - `welcome.mp3` - Welcome message
   - `final.mp3` - Final success message
   - For each riddle (1-5):
     - `riddle1_question.mp3` - First riddle question
     - `riddle1_hint.mp3` - First riddle hint
     - `riddle1_success.mp3` - First riddle success message
     - (and so on for riddles 2-5)

## Required Audio Files

```
audio/
├── welcome.mp3
├── final.mp3
├── riddle1_question.mp3
├── riddle1_hint.mp3
├── riddle1_success.mp3
├── riddle2_question.mp3
├── riddle2_hint.mp3
├── riddle2_success.mp3
├── riddle3_question.mp3
├── riddle3_hint.mp3
├── riddle3_success.mp3
├── riddle4_question.mp3
├── riddle4_hint.mp3
├── riddle4_success.mp3
├── riddle5_question.mp3
├── riddle5_hint.mp3
└── riddle5_success.mp3
```

## Dependencies

```bash
pip install opencv-python mediapipe numpy pygame
```

## Controls

- Make a "0" gesture (touch thumb and index finger) to repeat the current question
- Cross two index fingers to get a hint
- Show the correct number of fingers to answer:
  - Riddle 1: Three fingers (Firewall layers)
  - Riddle 2: Two fingers (Encryption keys)
  - Riddle 3: Four fingers (Authentication factors)
  - Riddle 4: One finger (Local backup)
  - Riddle 5: Five fingers (Cyber kill chain)

## Running the Game

```bash
python simple_gesture_puzzle.py
```

Press 'q' to quit at any time.

## Recording Tips for Gandalf-style Voice

When recording your audio files:
1. Use a deep, authoritative voice
2. Speak slowly and deliberately
3. Add dramatic pauses between sentences
4. Use a slight echo/reverb effect if possible
5. Add mystical background ambiance (optional)
6. Emphasize key words and numbers
7. End statements with a slight trailing off

## Audio File Content Suggestions

1. Welcome Message:
   "Ah... Welcome, brave seeker of digital wisdom! Before you lies the path of cyber enlightenment..."

2. Riddle Questions:
   - Start with a dramatic introduction
   - Speak the riddle in verse
   - End with a call to action

3. Hints:
   - Begin with "Mmm..." or "Ah..."
   - Speak thoughtfully and mysteriously
   - End with encouragement

4. Success Messages:
   - Start with excitement
   - Explain the answer
   - Transition to the next challenge

5. Final Message:
   - Build dramatic tension
   - Reveal the PIN with pauses between numbers
   - End with a grand statement 