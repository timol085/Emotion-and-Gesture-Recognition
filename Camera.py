import pandas as pd
import time
import numpy as np
from keras.models import load_model
import mediapipe as mp
import cv2
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = load_model('GestureModel.h5')

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape

analysisframe = ''
gesturePred = ["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace",
               "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"]
while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)

            # Perform analysis on the detected hand
            analysisframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            analysisframe = analysisframe[y_min:y_max, x_min:x_max]
            analysisframe = cv2.resize(analysisframe, (48, 48))

            nlist = []
            rows, cols = analysisframe.shape
            for i in range(rows):
                for j in range(cols):
                    k = analysisframe[i, j]
                    nlist.append(k)

            datan = pd.DataFrame(nlist).T
            colname = []
            for val in range(2304):
                colname.append(val)
            datan.columns = colname

            pixeldata = datan.values
            pixeldata = pixeldata / 255
            pixeldata = pixeldata.reshape(-1, 48, 48, 1)
            prediction = model.predict(pixeldata)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, gesturePred[maxindex], (x+5, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            predarray = np.array(prediction[0])
            letter_prediction_dict = {
                gesturePred[i]: predarray[i] for i in range(len(gesturePred))}
            predarrayordered = sorted(predarray, reverse=True)
            high1 = predarrayordered[0]
            high2 = predarrayordered[1]
            high3 = predarrayordered[2]
            for key, value in letter_prediction_dict.items():
                if value == high1:
                    print("Predicted Gesture 1: ", key)
                    print('Confidence 1: ', 100*value)
                elif value == high2:
                    print("Predicted Gesture 2: ", key)
                    print('Confidence 2: ', 100*value)
                elif value == high3:
                    print("Predicted Gesture 3: ", key)
                    print('Confidence 3: ', 100*value)

    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()
