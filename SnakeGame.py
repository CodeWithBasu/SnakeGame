import os
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

import cvzone
import cv2
import numpy as np
import math
import random
from cvzone.HandTrackingModule import HandDetector

# Setup openCV capture and window size      
capture = cv2.VideoCapture(0)
capture.set(3, 1280)  # Width
capture.set(4, 720)   # Height

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

class SnakeGame:
    def __init__(self, food_path):
        # Snake properties
        self.points = []  # List of snake points
        self.lengths = []  # Distances between points
        self.current_length = 0  # Total length of the snake
        self.allowed_length = 150  # Maximum allowed length
        self.previous_head = 640, 360  # Center of the screen as initial head position

        # Food properties
        self.food_img = cv2.imread(food_path, cv2.IMREAD_UNCHANGED)
        self.food_height, self.food_width, _ = self.food_img.shape
        self.food_location = 0, 0
        self.randomize_food_location()



        # Game properties
        self.score = 0
        self.game_over = False

    def randomize_food_location(self):
        """Randomly place the food within the game window."""
        self.food_location = random.randint(100, 1000), random.randint(100, 600)

    def update(self, img_main, current_head):
        """Update the game state."""
        if self.game_over:
            # Display game over screen
            cvzone.putTextRect(img_main, "Game Over", [300, 300], scale=7, thickness=5, colorT=(255, 255, 255),
                               colorR=(0, 0, 255), offset=20)
            cvzone.putTextRect(img_main, f"Score: {self.score}", [300, 450], scale=7, thickness=5, colorT=(255, 255, 255),
                               colorR=(0, 0, 255), offset=20)
        else:
            # Update snake's body
            previous_x, previous_y = self.previous_head
            current_x, current_y = current_head

            # Add new point to the snake
            self.points.append([current_x, current_y])
            distance = math.hypot(current_x - previous_x, current_y - previous_y)
            self.lengths.append(distance)
            self.current_length += distance
            self.previous_head = current_x, current_y

            # Reduce length if it exceeds the allowed length
            if self.current_length > self.allowed_length:
                for i, length in enumerate(self.lengths):
                    self.current_length -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.current_length < self.allowed_length:
                        break

            # Check if snake eats the food
            food_x, food_y = self.food_location
            if (food_x - self.food_width // 2 < current_x < food_x + self.food_width // 2 and
                    food_y - self.food_height // 2 < current_y < food_y + self.food_height // 2):
                self.randomize_food_location()
                self.allowed_length += 50
                self.score += 1
                print(f"Score: {self.score}")

            # Draw snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(img_main, self.points[i - 1], self.points[i], (0, 0, 255), 20)
                cv2.circle(img_main, self.points[-1], 20, (200, 0, 200), cv2.FILLED)

            # Draw food
            img_main = cvzone.overlayPNG(img_main, self.food_img,
                                         (food_x - self.food_width // 2, food_y - self.food_height // 2))

            # Display score
            cvzone.putTextRect(img_main, f"Score: {self.score}", [50, 80], scale=3, thickness=3, offset=10)

            # Collision detection
            if len(self.points) > 30:  # Increase minimum body length before checking collision
                pts = np.array(self.points[:-10], np.int32)  # Exclude the last 10 points for collision detection
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img_main, [pts], False, (0, 200, 0), 3)
                min_dist = cv2.pointPolygonTest(pts, (current_x, current_y), True)

                # Visualize collision area
                cv2.circle(img_main, (current_x, current_y), 10, (255, 255, 0), 2)

                # Debugging distance
                print(f"Min Dist: {min_dist}")

                # Check for collision
                if -5 <= min_dist <= 5:  # Loosened threshold
                    print("Collision detected! Game Over.")
                    self.game_over = True

        return img_main

# Initialize game
game = SnakeGame("Donut.png")

# Main game loop
while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally

    # Detect hands
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        # Get the landmark of the index finger tip
        lm_list = hands[0]['lmList']
        index_finger_tip = lm_list[8][0:2]
        img = game.update(img, index_finger_tip)

    # Display the image
    cv2.imshow("Snake Game", img)

    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord('r'):  # Restart game
        game.game_over = False
        game.score = 0
        game = SnakeGame("Donut.png")
    elif key == ord('q'):  # Quit game
        break

# Release resources
capture.release()
cv2.destroyAllWindows()