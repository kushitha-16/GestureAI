import cv2
import mediapipe as mp
import pyautogui
import math
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)

last_click_time = 0

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, frame = cap.read()

    if not success:
        print("Camera not detected")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip landmark = 8
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)

            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            # Convert webcam position to screen position
            screen_x = int(index_tip.x * screen_width)
            screen_y = int(index_tip.y * screen_height)

            pyautogui.moveTo(screen_x, screen_y)

            # Draw circle on index finger
            cv2.circle(frame, (index_x, index_y), 12, (0, 255, 0), cv2.FILLED)

            # Pinch detection
            pinch_distance = distance(
                (index_x, index_y),
                (thumb_x, thumb_y)
            )

            cv2.putText(
                frame,
                f"Pinch Distance: {int(pinch_distance)}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            current_time = time.time()

            if pinch_distance < 35 and current_time - last_click_time > 1:
                pyautogui.click()
                print("Left Click")
                last_click_time = current_time

                cv2.putText(
                    frame,
                    "LEFT CLICK",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )

    cv2.imshow("GestureAI Pro - Air Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()