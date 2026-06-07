import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

start_x = None
start_y = None
start_time = None

SWIPE_THRESHOLD = 80
TIME_LIMIT = 1.2
COOLDOWN = 1.5
last_action_time = 0


def perform_action(direction):
    if direction == "RIGHT":
        pyautogui.hotkey("ctrl", "tab")
        print("Right Swipe → Next Tab")

    elif direction == "LEFT":
        pyautogui.hotkey("ctrl", "shift", "tab")
        print("Left Swipe → Previous Tab")

    elif direction == "UP":
        pyautogui.scroll(5)
        print("Up Swipe → Scroll Up")

    elif direction == "DOWN":
        pyautogui.scroll(-5)
        print("Down Swipe → Scroll Down")


while True:
    success, frame = cap.read()

    if not success:
        print("Camera not detected")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    direction_text = "Show hand and swipe"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Use index fingertip for direction
            index_tip = hand_landmarks.landmark[8]

            current_x = int(index_tip.x * w)
            current_y = int(index_tip.y * h)

            cv2.circle(
                frame,
                (current_x, current_y),
                12,
                (0, 255, 0),
                cv2.FILLED
            )

            current_time = time.time()

            if start_x is None:
                start_x = current_x
                start_y = current_y
                start_time = current_time

            dx = current_x - start_x
            dy = current_y - start_y
            elapsed = current_time - start_time

            if elapsed <= TIME_LIMIT and current_time - last_action_time > COOLDOWN:

                if abs(dx) > abs(dy) and abs(dx) > SWIPE_THRESHOLD:

                    if dx > 0:
                        direction_text = "RIGHT SWIPE"
                        perform_action("RIGHT")

                    else:
                        direction_text = "LEFT SWIPE"
                        perform_action("LEFT")

                    last_action_time = current_time
                    start_x = None
                    start_y = None
                    start_time = None

                elif abs(dy) > abs(dx) and abs(dy) > SWIPE_THRESHOLD:

                    if dy < 0:
                        direction_text = "UP SWIPE"
                        perform_action("UP")

                    else:
                        direction_text = "DOWN SWIPE"
                        perform_action("DOWN")

                    last_action_time = current_time
                    start_x = None
                    start_y = None
                    start_time = None

            if elapsed > TIME_LIMIT:
                start_x = current_x
                start_y = current_y
                start_time = current_time

            cv2.putText(
                frame,
                f"Direction: {direction_text}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Use INDEX FINGER to swipe",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

    else:
        start_x = None
        start_y = None
        start_time = None

    cv2.imshow("GestureAI Pro - Direction Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()