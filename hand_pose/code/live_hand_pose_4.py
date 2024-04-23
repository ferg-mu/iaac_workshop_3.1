import cv2
import mediapipe as mp
import math
import numpy as np
from pythonosc import udp_client
from pythonosc import osc_message_builder


# min & max values for normalisation
# area should aim to be 
MAX_AREA = 20000  # this should be set to a plausible maximum area
MAX_LENGTH = 300  # this should be set to a plausible maximum length for any line between fingers

ip = "192.168.1.138"
port = 55556



class HandProcessor:
    def __init__(self, landmarks, frame_shape, handedness, client):
        self.landmarks = landmarks
        self.frame_shape = frame_shape
        self.handedness = handedness
        self.thumb_tip_pixel = (0, 0)
        self.index_finger_tip_pixel = (0, 0)
        self.pinky_tip_pixel = (0, 0)
        self.middle_finger_tip_pixel = (0, 0)
        self.ring_finger_tip_pixel = (0, 0)
        self.wrist_tip_pixel = (0, 0)
        self.index_line_length = 0
        self.middle_line_length = 0
        self.pinky_line_length = 0
        self.ring_line_length = 0
        self.max_circle = None
        self.calculate_metrics()
        self.values = []
        self.client = client
        self.x_pos = 0
        self.y_pos = 0

    def calculate_metrics(self):
        thumb_tip = self.landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_finger_tip = self.landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = self.landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
        middle_finger_tip = self.landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = self.landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        wrist = self.landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

        # Convert normalized coordinates to pixel coordinates as NumPy arrays
        self.thumb_tip_pixel = np.array([thumb_tip.x, thumb_tip.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])
        self.index_finger_tip_pixel = np.array([index_finger_tip.x, index_finger_tip.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])
        self.pinky_tip_pixel = np.array([pinky_tip.x, pinky_tip.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])
        self.middle_finger_tip_pixel = np.array([middle_finger_tip.x, middle_finger_tip.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])
        self.ring_finger_tip_pixel = np.array([ring_finger_tip.x, ring_finger_tip.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])
        self.wrist_tip_pixel = np.array([wrist.x, wrist.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])

        points = np.array([[lm.x * self.frame_shape[1], lm.y * self.frame_shape[0]] for lm in self.landmarks.landmark]).astype(int)
        center, radius = cv2.minEnclosingCircle(points)
        self.max_circle = (center, radius)
        self.area = math.pi * radius ** 2

        # Calculate distances using NumPy for element-wise subtraction
        self.index_line_length = np.linalg.norm(self.index_finger_tip_pixel - self.thumb_tip_pixel)
        self.pinky_line_length = np.linalg.norm(self.pinky_tip_pixel - self.thumb_tip_pixel)
        self.middle_line_length = np.linalg.norm(self.middle_finger_tip_pixel - self.thumb_tip_pixel)
        self.ring_line_length = np.linalg.norm(self.ring_finger_tip_pixel - self.thumb_tip_pixel)

        # Calculate normalized center coordinates
        self.x_pos = center[0] / self.frame_shape[1]
        self.y_pos = center[1] / self.frame_shape[0]

        delta_x = center[0] - self.wrist_tip_pixel[0]
        delta_y = center[1] - self.wrist_tip_pixel[1]
        angle_from_horizontal = math.degrees(math.atan2(delta_y, delta_x))
        self.wrist_to_center_angle = (angle_from_horizontal + 90) % 360

        self.index_line_length = np.linalg.norm(np.array([index_finger_tip.x, index_finger_tip.y]) * self.frame_shape[:2] - np.array([thumb_tip.x, thumb_tip.y]) * self.frame_shape[:2])
        self.pinky_line_length = np.linalg.norm(self.pinky_tip_pixel - self.thumb_tip_pixel)
        self.middle_line_length = np.linalg.norm(self.middle_finger_tip_pixel - self.thumb_tip_pixel)
        self.ring_line_length = np.linalg.norm(self.ring_finger_tip_pixel - self.thumb_tip_pixel)

    def print_metrics(self):
        normalized_area = (self.area / MAX_AREA)
        normalized_index_length = self.index_line_length / MAX_LENGTH
        normalized_pinky_length = self.pinky_line_length / MAX_LENGTH
        normalized_middle_length = self.middle_line_length / MAX_LENGTH
        normalized_ring_length = self.ring_line_length / MAX_LENGTH
        angle_radians = math.radians(self.wrist_to_center_angle - 90)
        normalized_angle = math.cos(angle_radians)

        print(f"{self.handedness.capitalize()} hand: Area: {normalized_area:.3f}, Index to Thumb: {normalized_index_length:.3f}, "
              f"Middle to Thumb: {normalized_middle_length:.3f}, Pinky to Thumb: {normalized_pinky_length:.3f}, Ring to Thumb: {normalized_ring_length:.3f}, "
              f"Wrist to Center Angle: {normalized_angle:.3f}, x_pos: {self.x_pos:.3f}, y_pos: {self.y_pos:.3f}")

        # Send OSC messages
        self.client.send_message("/hand/metrics", [self.handedness, normalized_area, normalized_index_length, self.x_pos, self.y_pos, normalized_angle])

        self.values.append(self.area)
        if len(self.values) == 10:
            avg = sum(self.values) / len(self.values)
            normalized_avg_area = avg / MAX_AREA
            print(f"Average area over last 10 values: {normalized_avg_area:.3f}")
            self.values.clear()
        
        
        index = 1
        address = "/motion/rot"
        rx = normalized_index_length
        ry = 0
        rz = 0
        message = [index, rx, ry, rz]
        client.send_message(address, message)
        print(f"Sent message: {address}/{message}")

        address = "/motion/pos"
        x = 0.75 #angle_radians
        y = 0.75 #normalized_middle_length
        z = 0.75 #normalized_area

        message = [index, x, y, z]
        client.send_message(address, message)
        print(f"Sent message: {address}/{message}")
        # time.sleep(0.01)

    def draw_hand(self, image):
        if self.max_circle:
            # Draw the enclosing circle
            cv2.circle(image, (int(self.max_circle[0][0]), int(self.max_circle[0][1])), int(self.max_circle[1]), (255, 255, 255), 2)

        # Draw hand skeleton
        mp.solutions.drawing_utils.draw_landmarks(
            image, self.landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=1),
            mp.solutions.drawing_utils.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=1)
        )

        # Draw lines between fingers, ensure coordinates are integer tuples
        cv2.line(image, tuple(self.thumb_tip_pixel.astype(int)), tuple(self.index_finger_tip_pixel.astype(int)), (0, 255, 0), 2)
        cv2.line(image, tuple(self.thumb_tip_pixel.astype(int)), tuple(self.pinky_tip_pixel.astype(int)), (0, 255, 255), 2)
        cv2.line(image, tuple(self.thumb_tip_pixel.astype(int)), tuple(self.middle_finger_tip_pixel.astype(int)), (0, 0, 255), 2)
        cv2.line(image, tuple(self.thumb_tip_pixel.astype(int)), tuple(self.ring_finger_tip_pixel.astype(int)), (255, 0, 0), 2)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)
client = udp_client.SimpleUDPClient(ip, port)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    black_image = np.zeros(frame.shape, dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            hand_processor = HandProcessor(hand_landmarks, frame.shape[:2], hand_label, client)
            hand_processor.draw_hand(black_image)
            hand_processor.print_metrics()

    cv2.imshow('Hand Pose', black_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()