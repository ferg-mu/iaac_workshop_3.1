import cv2
import mediapipe as mp
import serial
import math
import pickle
import socket
import numpy as np
import csv
import os

# Open the serial port
ser = serial.Serial('COM3', 9600) 

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenCV video capture
cap = cv2.VideoCapture(1)
# Initialize an empty list to store the area values
areas = []

while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    # Create a black image of the same size as the frame
    black_image = np.zeros(frame.shape, dtype=np.uint8)

    # Set default values
    average_area = 500
    x = 0
    y = 0
    index_line_length = 31
    pinky_line_length = 31
    middle_line_length = 31

    # If hand landmarks are detected, draw them on the frame
    if results.multi_hand_landmarks:
        max_area = 0
        max_circle = None
        for hand_landmarks in results.multi_hand_landmarks:
            points = []
            for landmark in hand_landmarks.landmark:
                # Get the coordinates of the thumb tip and index finger tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Convert the normalized landmark coordinates to pixel coordinates
                thumb_tip_pixel = np.multiply([thumb_tip.x, thumb_tip.y], [frame.shape[1], frame.shape[0]]).astype(int)
                index_finger_tip_pixel = np.multiply([index_finger_tip.x, index_finger_tip.y], [frame.shape[1], frame.shape[0]]).astype(int)
                pinky_tip_pixel = np.multiply([pinky_tip.x, pinky_tip.y], [frame.shape[1], frame.shape[0]]).astype(int)
                middle_finger_tip_pixel = np.multiply([middle_finger_tip.x, middle_finger_tip.y], [frame.shape[1], frame.shape[0]]).astype(int)
                # calculate length of the index line
                index_line_length = math.sqrt((index_finger_tip_pixel[0] - thumb_tip_pixel[0]) ** 2 + (index_finger_tip_pixel[1] - thumb_tip_pixel[1]) ** 2)
                if index_line_length < 30:
                    index_line_length = 0
                points.append([int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])])
                # calculate the length of the line between the thumb and pinky finger
                pinky_line_length = math.sqrt((pinky_tip_pixel[0] - thumb_tip_pixel[0]) ** 2 + (pinky_tip_pixel[1] - thumb_tip_pixel[1]) ** 2)
                if pinky_line_length < 30:
                    pinky_line_length = 0
                # calculate the length of the line between the thumb and middle finger
                middle_line_length = math.sqrt((middle_finger_tip_pixel[0] - thumb_tip_pixel[0]) ** 2 + (middle_finger_tip_pixel[1] - thumb_tip_pixel[1]) ** 2)
                if middle_line_length < 30:
                    middle_line_length = 0
                

            # Find the minimum enclosing circle
            center, radius = cv2.minEnclosingCircle(np.array(points))

            # Calculate the area of the circle
            area = math.pi * radius ** 2

            # If this circle has a larger area, update max_area and max_circle
            if area > max_area:
                max_area = area
                max_circle = (center, radius)

        # If a circle was found, draw the circle with the largest area
        if max_circle is not None:
            cv2.circle(black_image, (int(max_circle[0][0]), int(max_circle[0][1])), int(max_circle[1]), (255, 255, 255), 2)
            # calculate the area of the circle
            areas.append(max_area)
            # filter the areas to get the last 10 values and print the average
            if len(areas) > 10:
                areas.pop(0)

            # compute the x and y coordinates of the center of the circle
            x, y = max_circle[0]
            average_area = round(sum(areas) / len(areas), 2)

        mp.solutions.drawing_utils.draw_landmarks(black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Draw a line from the thumb tip to the index finger tip
        cv2.line(black_image, tuple(thumb_tip_pixel), tuple(index_finger_tip_pixel), (0, 255, 0), 1)
        # Draw a line from the thumb tip to the pinky finger tip
        cv2.line(black_image, tuple(thumb_tip_pixel), tuple(pinky_tip_pixel), (0, 255, 255), 1)
        # Draw a line from the thumb tip to the middle finger tip
        cv2.line(black_image, tuple(thumb_tip_pixel), tuple(middle_finger_tip_pixel), (0, 0, 255), 1)

    # print the x and y coordinates of the center of the circle
    print("Avg ar:", average_area, "x:", round(x, 2), "y:", round(y, 2), "T_to_I:", round(index_line_length, 2), "T_to_P:", round(pinky_line_length, 2), "T_to_M:", round(middle_line_length, 2))

    # Write the values to the serial port
    print("Average area:", average_area, "x:", round(x, 2), "y:", round(y, 2), "Thumb to Index:", round(index_line_length, 2), "Pinky to Thumb:", round(pinky_line_length, 2), "Thumb to Middle:", round(middle_line_length, 2))

    # Show video
    cv2.imshow('Hand Pose', black_image)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()