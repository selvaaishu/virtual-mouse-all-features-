import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

click_down = False
prev_click_time = 0

def fingers_up(lmList):
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    fingers = []

    # Thumb
    fingers.append(lmList[4].x < lmList[3].x)

    # Other fingers
    for id in tips:
        fingers.append(lmList[id].y < lmList[id - 2].y)

    return fingers

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = handLms.landmark
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Get index finger tip coordinates
            x = int(lmList[8].x * 640)
            y = int(lmList[8].y * 480)
            screen_x = np.interp(x, (0, 640), (0, screen_w))
            screen_y = np.interp(y, (0, 480), (0, screen_h))
            pyautogui.moveTo(screen_x, screen_y)

            # Gesture recognition
            fingerStates = fingers_up(lmList)

            # Left click gesture (index + thumb close together)
            thumb_tip = lmList[4]
            index_tip = lmList[8]
            distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

            if distance < 0.04:
                if time.time() - prev_click_time > 1:
                    pyautogui.click()
                    prev_click_time = time.time()
                    print("Left Click")

            # Right click (index + middle up)
            if fingerStates[1] and fingerStates[2] and not any(fingerStates[3:]):
                pyautogui.rightClick()
                print("Right Click")
                time.sleep(1)

            # Scroll up (thumb only)
            if fingerStates[0] and not any(fingerStates[1:]):
                pyautogui.scroll(20)
                print("Scroll Up")

            # Scroll down (pinky only)
            if fingerStates[4] and not any(fingerStates[:4]):
                pyautogui.scroll(-20)
                print("Scroll Down")

    cv2.imshow("Virtual Mouse - All Features", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
