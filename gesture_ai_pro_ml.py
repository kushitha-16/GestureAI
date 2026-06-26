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
import warnings

warnings.filterwarnings("ignore")

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
    global current_action

    if command == "":
        return

    if "open chrome" in command:
        current_action = "Opening Chrome"
        speak("Opening Chrome")
        os.system("start chrome")

    elif "open youtube" in command:
        current_action = "Opening YouTube"
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com")

    elif "open github" in command:
        current_action = "Opening GitHub"
        speak("Opening GitHub")
        webbrowser.open("https://github.com")

    elif "open vs code" in command or "open visual studio code" in command:
        current_action = "Opening VS Code"
        speak("Opening VS Code")
        os.system("code")

    elif "search" in command:
        query = command.replace("search", "").strip()
        current_action = f"Searching {query}"
        speak(f"Searching for {query}")
        webbrowser.open(f"https://www.google.com/search?q={query}")

    elif "volume up" in command:
        current_action = "Volume Up"
        speak("Increasing volume")
        pyautogui.press("volumeup")

    elif "volume down" in command:
        current_action = "Volume Down"
        speak("Decreasing volume")
        pyautogui.press("volumedown")

    elif "play" in command or "pause" in command:
        current_action = "Play / Pause"
        speak("Playing or pausing media")
        pyautogui.press("playpause")

    else:
        current_action = "Unknown voice command"
        speak("Command not recognized yet")


MODEL_FILE = "gesture_model.pkl"
model = joblib.load(MODEL_FILE)

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
current_action = "Waiting..."
prev_time = 0

ACTION_COOLDOWN = 4

smooth_x = 0
smooth_y = 0
SMOOTHING = 7
FRAME_REDUCTION = 80

CONFIDENCE_THRESHOLD = 75
STABLE_FRAME_COUNT = 5
gesture_history = []
stable_gesture = "No Hand"


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

    return fingers_up == [1, 1, 1, 0]

def get_stable_gesture(predicted_gesture, confidence):
    global gesture_history, stable_gesture

    if confidence < CONFIDENCE_THRESHOLD:
        gesture_history = []
        stable_gesture = "Low Confidence"
        return stable_gesture

    gesture_history.append(predicted_gesture)

    if len(gesture_history) > STABLE_FRAME_COUNT:
        gesture_history.pop(0)

    if len(gesture_history) == STABLE_FRAME_COUNT:
        if all(g == predicted_gesture for g in gesture_history):
            stable_gesture = predicted_gesture

    return stable_gesture

def air_mouse_control(hand_landmarks, frame_width, frame_height):
    global last_click_time, smooth_x, smooth_y, current_action

    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]

    index_x = int(index_tip.x * frame_width)
    index_y = int(index_tip.y * frame_height)

    thumb_x = int(thumb_tip.x * frame_width)
    thumb_y = int(thumb_tip.y * frame_height)

    x = max(FRAME_REDUCTION, min(frame_width - FRAME_REDUCTION, index_x))
    y = max(FRAME_REDUCTION, min(frame_height - FRAME_REDUCTION, index_y))

    screen_x = (x - FRAME_REDUCTION) * screen_width / (frame_width - 2 * FRAME_REDUCTION)
    screen_y = (y - FRAME_REDUCTION) * screen_height / (frame_height - 2 * FRAME_REDUCTION)

    smooth_x = smooth_x + (screen_x - smooth_x) / SMOOTHING
    smooth_y = smooth_y + (screen_y - smooth_y) / SMOOTHING

    pyautogui.moveTo(int(smooth_x), int(smooth_y), duration=0.01)

    pinch_distance = distance((index_x, index_y), (thumb_x, thumb_y))
    current_time = time.time()

    if pinch_distance < 30 and current_time - last_click_time > 1:
        pyautogui.click()
        current_action = "Left Click"
        print("Left Click")
        last_click_time = current_time

    return index_x, index_y, pinch_distance

def format_gesture_name(gesture):
    names = {
        "palm": "Palm",
        "victory": "Victory",
        "one_finger": "One Finger",
        "ok_sign": "OK Sign",
        "thumbs_up": "Thumbs Up",
        "three_fingers": "Three Fingers",
        "rock": "Rock",
        "call_me": "Call Me",
        "peace": "Peace",
        "fist": "Fist",
        "stop": "Stop",
        "No Hand": "No Hand",
        "Low Confidence": "Low Confidence"
    }
    return names.get(gesture, gesture)

def draw_dashboard(frame, gesture, action, status, confidence, fps):
    cv2.rectangle(frame, (0, 0), (700, 220), (35, 35, 35), -1)

    cv2.putText(frame, "GestureAI Pro v2.0", (20, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

    cv2.line(frame, (20, 50), (620, 50), (100, 100, 100), 2)

    cv2.putText(frame, f"Gesture     : {format_gesture_name(gesture)}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame, f"Action      : {action}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    status_color = (0, 255, 0) if status == "Tracking" else (0, 0, 255)

    cv2.putText(frame, f"Status      : {status}", (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.putText(frame, f"Confidence  : {confidence:.1f}%", (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"FPS : {fps}", (500, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def perform_ml_action(gesture, hand_landmarks, frame_width, frame_height):
    global last_action_time, ready_for_palm, current_action

    current_time = time.time()

    if gesture == "fist":
        current_action = "Exit Application"
        speak("Goodbye")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    if gesture == "one_finger":
        current_action = "Air Mouse Active"
        index_x, index_y, pinch_distance = air_mouse_control(
            hand_landmarks,
            frame_width,
            frame_height
        )
        return "Air Mouse Active", index_x, index_y, pinch_distance

    if current_time - last_action_time < ACTION_COOLDOWN:
        return None, None, None, None

    if gesture == "palm" and ready_for_palm:
        current_action = "Voice Assistant"
        command = listen_command()
        execute_voice_command(command)
        ready_for_palm = False
        last_action_time = current_time

    elif gesture == "victory":
        current_action = "Opening VS Code"
        speak("Opening VS Code")
        os.system("code")
        last_action_time = current_time

    elif gesture == "thumbs_up":
        current_action = "Volume Up"
        speak("Volume Up")
        pyautogui.press("volumeup")
        last_action_time = current_time

    elif gesture == "three_fingers":
        current_action = "Volume Down"
        speak("Volume Down")
        pyautogui.press("volumedown")
        last_action_time = current_time

    elif gesture == "rock":
        current_action = "Play / Pause"
        speak("Play or Pause")
        pyautogui.press("playpause")
        last_action_time = current_time

    elif gesture == "call_me":
        current_action = "Opening YouTube"
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com")
        last_action_time = current_time

    elif gesture == "peace":
        current_action = "Opening GitHub"
        speak("Opening GitHub")
        webbrowser.open("https://github.com")
        last_action_time = current_time

    elif gesture == "ok_sign":
        current_action = "Left Click"
        speak("Left Click")
        pyautogui.click()
        last_action_time = current_time

    elif gesture == "stop":
        current_action = "Stop Detected"
        speak("Stop gesture detected")
        last_action_time = current_time

    return None, None, None, None


while True:
    success, frame = cap.read()

    if not success:
        print("Camera not detected")
        break

    current_time_for_fps = time.time()
    fps = int(1 / (current_time_for_fps - prev_time)) if prev_time != 0 else 0
    prev_time = current_time_for_fps

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    predicted_gesture = "No Hand"
    display_gesture = "No Hand"
    confidence = 0.0
    status = "Waiting"

    if result.multi_hand_landmarks:
        status = "Tracking"

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = extract_landmarks(hand_landmarks)

            prob = model.predict_proba([landmarks])[0]
            confidence = max(prob) * 100
            raw_prediction = model.classes_[prob.argmax()]

            if detect_three_fingers_rule(hand_landmarks):
                predicted_gesture = "three_fingers"
                confidence = max(confidence, 95.0)
            else:
                predicted_gesture = raw_prediction

            stable_prediction = get_stable_gesture(predicted_gesture, confidence)
            display_gesture = stable_prediction

            if stable_prediction not in ["Low Confidence", "No Hand"]:
                action_text, index_x, index_y, pinch_distance = perform_ml_action(
                    stable_prediction,
                    hand_landmarks,
                    w,
                    h
                )
            else:
                action_text, index_x, index_y, pinch_distance = None, None, None, None

            if index_x is not None:
                cv2.circle(frame, (index_x, index_y), 15, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (index_x, index_y), 25, (255, 255, 255), 2)

                cv2.putText(
                    frame,
                    f"Pinch: {int(pinch_distance)}",
                    (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

    else:
        ready_for_palm = True
        current_action = "Waiting..."
        gesture_history = []
        stable_gesture = "No Hand"
        display_gesture = "No Hand"

    cv2.rectangle(
        frame,
        (FRAME_REDUCTION, FRAME_REDUCTION),
        (w - FRAME_REDUCTION, h - FRAME_REDUCTION),
        (255, 255, 0),
        2
    )

    draw_dashboard(
        frame,
        display_gesture,
        current_action,
        status,
        confidence,
        fps
    )

    cv2.imshow("GestureAI Pro Dashboard", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()