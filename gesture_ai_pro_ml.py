import cv2
import mediapipe as mp
import pyautogui
import os
import time
import math
import joblib
import speech_recognition as sr
import pyttsx3
import webbrowser

# =========================
# Voice setup
# =========================

engine = pyttsx3.init()

def speak(text):
    print("Assistant:", text)
    engine.say(text[:250])
    engine.runAndWait()

recognizer = sr.Recognizer()

def listen_command():
    with sr.Microphone() as source:
        speak("Listening for your command")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=8)
            command = recognizer.recognize_google(audio).lower()
            print("You said:", command)
            return command

        except sr.WaitTimeoutError:
            speak("No voice detected")
            return ""

        except sr.UnknownValueError:
            speak("Sorry, I could not understand")
            return ""

        except sr.RequestError:
            speak("Speech service error")
            return ""

def execute_voice_command(command):
    if command == "":
        return

    if "open chrome" in command:
        speak("Opening Chrome")
        os.system("start chrome")

    elif "open youtube" in command:
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com")

    elif "open github" in command:
        speak("Opening GitHub")
        webbrowser.open("https://github.com")

    elif "open vs code" in command or "open visual studio code" in command:
        speak("Opening VS Code")
        os.system("code")

    elif "search" in command:
        query = command.replace("search", "").strip()
        speak(f"Searching for {query}")
        webbrowser.open(f"https://www.google.com/search?q={query}")

    elif "volume up" in command:
        speak("Increasing volume")
        pyautogui.press("volumeup")

    elif "volume down" in command:
        speak("Decreasing volume")
        pyautogui.press("volumedown")

    elif "play" in command or "pause" in command:
        speak("Playing or pausing media")
        pyautogui.press("playpause")

    else:
        speak("Command not recognized yet")

# =========================
# Load ML model
# =========================

MODEL_FILE = "gesture_model.pkl"
model = joblib.load(MODEL_FILE)

# =========================
# MediaPipe setup
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

last_action_time = 0
last_click_time = 0
ready_for_palm = True

ACTION_COOLDOWN = 4

# =========================
# Helper functions
# =========================

def extract_landmarks(hand_landmarks):
    data = []

    for landmark in hand_landmarks.landmark:
        data.extend([landmark.x, landmark.y, landmark.z])

    return data

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def detect_three_fingers_rule(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    fingers_up = []

    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

    # Index + middle + ring up, pinky down
    return fingers_up == [1, 1, 1, 0]

def air_mouse_control(hand_landmarks, frame_width, frame_height):
    global last_click_time

    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]

    index_x = int(index_tip.x * frame_width)
    index_y = int(index_tip.y * frame_height)

    thumb_x = int(thumb_tip.x * frame_width)
    thumb_y = int(thumb_tip.y * frame_height)

    screen_x = int(index_tip.x * screen_width)
    screen_y = int(index_tip.y * screen_height)

    pyautogui.moveTo(screen_x, screen_y)

    pinch_distance = distance(
        (index_x, index_y),
        (thumb_x, thumb_y)
    )

    current_time = time.time()

    if pinch_distance < 35 and current_time - last_click_time > 1:
        pyautogui.click()
        print("Left Click")
        last_click_time = current_time

    return index_x, index_y, pinch_distance

def perform_ml_action(gesture, hand_landmarks, frame_width, frame_height):
    global last_action_time, ready_for_palm

    current_time = time.time()

    if gesture == "fist":
        speak("Goodbye")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    if gesture == "one_finger":
        index_x, index_y, pinch_distance = air_mouse_control(
            hand_landmarks,
            frame_width,
            frame_height
        )
        return "Air Mouse Active", index_x, index_y, pinch_distance

    if current_time - last_action_time < ACTION_COOLDOWN:
        return None, None, None, None

    if gesture == "palm" and ready_for_palm:
        command = listen_command()
        execute_voice_command(command)
        ready_for_palm = False
        last_action_time = current_time

    elif gesture == "victory":
        speak("Opening VS Code")
        os.system("code")
        last_action_time = current_time

    elif gesture == "thumbs_up":
        speak("Volume Up")
        pyautogui.press("volumeup")
        last_action_time = current_time

    elif gesture == "three_fingers":
        speak("Volume Down")
        pyautogui.press("volumedown")
        last_action_time = current_time

    elif gesture == "rock":
        speak("Play or Pause")
        pyautogui.press("playpause")
        last_action_time = current_time

    elif gesture == "call_me":
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com")
        last_action_time = current_time

    elif gesture == "peace":
        speak("Opening GitHub")
        webbrowser.open("https://github.com")
        last_action_time = current_time

    elif gesture == "ok_sign":
        speak("Left Click")
        pyautogui.click()
        last_action_time = current_time

    elif gesture == "stop":
        speak("Stop gesture detected")
        last_action_time = current_time

    return None, None, None, None

# =========================
# Main loop
# =========================

while True:
    success, frame = cap.read()

    if not success:
        print("Camera not detected")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    predicted_gesture = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = extract_landmarks(hand_landmarks)

            raw_prediction = model.predict([landmarks])[0]

            if detect_three_fingers_rule(hand_landmarks):
                predicted_gesture = "three_fingers"
            else:
                predicted_gesture = raw_prediction

            action_text, index_x, index_y, pinch_distance = perform_ml_action(
                predicted_gesture,
                hand_landmarks,
                w,
                h
            )

            cv2.putText(
                frame,
                f"ML Gesture: {predicted_gesture}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            if action_text:
                cv2.putText(
                    frame,
                    action_text,
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2
                )

            if index_x is not None:
                cv2.circle(
                    frame,
                    (index_x, index_y),
                    12,
                    (0, 255, 0),
                    cv2.FILLED
                )

                cv2.putText(
                    frame,
                    f"Pinch: {int(pinch_distance)}",
                    (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

    else:
        ready_for_palm = True

        cv2.putText(
            frame,
            "No Hand",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("GestureAI Pro ML", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()