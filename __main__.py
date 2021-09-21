'''
README:
# gesture-detection

This project has been built in collaboration with Anshul Singh (https://www.github.com/AnshulXing)

## Setup:

Install cv2, mediapipe and numpy:

Use pip(for python2) or pip3(for python3) to install both the libraries

Commands:<br>
For open-cv: ```pip3 install opencv-python```<br>
For mediapipe: ```pip3 install mediapipe```<br>
For numpy: ```pip3 install numpy```<br>

### For Windows systems, it needs nircmd installed additionally. <br>

Download this zip: http://www.nirsoft.net/utils/nircmd.zip and then copy the nircmd.exe file to C:\\Windows:
'''

import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import os
import sys

cap = cv.VideoCapture(0) 

# **THIS DOESNT WORK FOR NOW BUT DOESN'T CREATE PROBLEMS EITHER
cam_width, cam_height = 1280, 800
cap.set(3, cam_width)
cap.set(4, cam_height)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 1, min_detection_confidence = 0.75)

mp_draw = mp.solutions.drawing_utils

previous_time = 0 # setting previous time at 0 initially

# Calibrating the distance-volume relation 
distance_at_max_volume = 270
distance_at_min_volume = 25

while True:

    success, img = cap.read()
    img_RGB = cv.cvtColor(img,  cv.COLOR_BGR2RGB) # Getting a RGB version of image to pass in the hands.process
    results = hands.process(img_RGB) # This will return the results after capturing the hands from the image

    if results.multi_hand_landmarks: # Checking if it is not 'None' type
        for hand_landmark in results.multi_hand_landmarks: # Iterating over all the hands captured in the image
            
            index_finger = hand_landmark.landmark[8] #Check documentation of mediapipe to know how 8 and 4 represent index and thumb
            thumb = hand_landmark.landmark[4]

            h,w,_ = img.shape # Getting height and width of the image window

            distance = abs(thumb.x - index_finger.x), abs(thumb.y - index_finger.y) #extracted distances in x and y as tuple (x,y)
            # distance  = (x, y)

            euclidean = np.sqrt((distance[0]*w)**2 + (distance[1]*h)**2) # Baudhayana theorem, multiplied by h and w to scale it to the window
            print(f'Distance: {euclidean}')
            cv.putText(img, f'Distance: {euclidean}', (10, 60), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

            volume_status = euclidean/(distance_at_max_volume - distance_at_min_volume) * 100 # Calculated Volume distance
            if volume_status>100: # This is to make sure the volume percentage never exceeds 100 and goes below 0
                volume_status = 100
            if volume_status<0:
                volume_status = 0

            cv.putText(img, f'Volume: {int(volume_status)}%', (500, 90), cv.FONT_HERSHEY_PLAIN,1,(0, 0, 255), 1)
            if sys.platform == 'linux' or sys.platform == 'linux2':
                os.system(f"amixer -D pulse sset Master {volume_status}%") # This actually changes the volume of the system
            elif sys.platform == 'win32':
                os.system(f'nircmd.exe setsysvolume {volume_status/100 * 65535}')
            
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS) # Draws the connections between landmarks

    current_time = time.time()
    fps = 1/(current_time-previous_time) # Calculating Frames per second rate
    previous_time = current_time

    cv.putText(img, f'FPS: {int(fps)}', (10,30), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

    cv.imshow('Image', img)
    cv.waitKey(1)
