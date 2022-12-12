#Rainier Jericho Lugtu
#Aldrin Macatangay
#CS - 302

import cv2 # for computer vision
import numpy as npy
import mediapipe as mpipe # for recognizing hand keypoints
import tensorflow as tf # use its pretrained model
from tensorflow.keras.models import load_model

# initializing mediapipe
mpipeHands = mpipe.solutions.hands # this performs the hand recognition
hands = mpipeHands.Hands(max_num_hands=1, min_detection_confidence=0.7) # only one hand for each frame
mpipeDraw = mpipe.solutions.drawing_utils # to automatically draw key points of the hand


# to load the pretrained model from tensorflow
model = load_model('mp_hand_gesture') #for pre-trained model
# Loading the classnames
# gestures are ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']
f = open('gesture.names', 'r') # first parameter is a file that has the 10 gesture classes' name
classNames = f.read().split('\n') # open the file
f.close()
print(classNames)




# Initializing web camera
capture = cv2.VideoCapture(0) # 0 is the ID of the webcam




while True:
    # Read each frame from the webcam
    _, frame = capture.read() # reads web camera's frame
    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1) # this mirrors the frame
    frameToRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert the frames from BGR to RGB format

    # Get hand landmark prediction
    result = hands.process(frameToRGB) # returned class

    # print(result)
  
    className = ''

    # post process of output
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks: # verify if there is a hand in the frame
            for lm in handslms.landmark: # loop thru the detections and keep the coords on landmarks
                # print(id, lm)
                lmx = int(lm.x * x) 
                lmy = int(lm.y * y)
                # product of height (y) and width (x) will be multiplied with the product. Value in the result will be 0 or 1
                landmarks.append([lmx, lmy]) 
            mpipeDraw.draw_landmarks(frame, handslms, mpipeHands.HAND_CONNECTIONS,
            mpipeDraw.DrawingSpec(color=(56,172,236), thickness=2, circle_radius=2),
            mpipeDraw.DrawingSpec(color=(216,31,42), thickness=2, circle_radius=2)) # to draw landmarks on the frame

            # Gesture prediction
            prediction = model.predict([landmarks]) # Grabs a list of landmarks then it returns an array that has 10 classes for prediction /landmark
            # print(prediction)
            classID = npy.argmax(prediction) # this return the max val fromt the list
            className = classNames[classID] # take classname after taking the index


    # Display prediction
    cv2.putText(frame, className, (25, 450),
               cv2.FONT_HERSHEY_COMPLEX, 1, (0,176,24), 2, cv2.LINE_AA) # display detected gesture


    # Display 
    cv2.imshow("Lugtu_Macatangay", frame) 


    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):
        break


# Quit
capture.release()
cv2.destroyAllWindows()