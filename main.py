import cv2
import mediapipe as mp
import pyautogui
import os
import time
import math
import speech_recognition as sr
import pyttsx3
import webbrowser

# Voice setup
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

# MediaPipe setup
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

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def count_fingers(hand_landmarks):
    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index, middle, ring, pinky
    for tip in [8, 12, 16, 20]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def recognize_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"

    elif fingers == [1, 1, 1, 1, 1]:
        return "Palm"

    elif fingers == [0, 1, 1, 0, 0]:
        return "Victory"

    elif fingers == [0, 1, 0, 0, 0]:
        return "Air Mouse"

    elif fingers == [0, 1, 1, 1, 0]:
        return "Three Fingers"

    return "Unknown"

def perform_gesture_action(gesture):
    if gesture == "Palm":
        command = listen_command()
        execute_voice_command(command)

    elif gesture == "Victory":
        speak("Opening VS Code")
        os.system("code")

    elif gesture == "Three Fingers":
        speak("Volume Down")
        pyautogui.press("volumedown")

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

while True:
    success, frame = cap.read()

    if not success:
        print("Camera not detected")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    gesture = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            fingers = count_fingers(hand_landmarks)
            gesture = recognize_gesture(fingers)

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            current_time = time.time()

            if gesture == "Fist":
                speak("Goodbye")
                cap.release()
                cv2.destroyAllWindows()
                exit()

            if gesture == "Air Mouse":
                index_x, index_y, pinch_distance = air_mouse_control(
                    hand_landmarks,
                    w,
                    h
                )

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
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

            elif gesture == "Palm" and ready_for_palm:
                if current_time - last_action_time > 2:
                    perform_gesture_action(gesture)
                    last_action_time = current_time
                    ready_for_palm = False

            elif gesture in ["Victory", "Three Fingers"]:
                if current_time - last_action_time > 4:
                    perform_gesture_action(gesture)
                    last_action_time = current_time

    else:
        ready_for_palm = True

    cv2.imshow("GestureAI Pro - Combined Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()