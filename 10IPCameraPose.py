import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imutils

# Initialize the initial position as the origin
current_position = np.zeros((3, 1), dtype=np.float32)

# Initialize the previous position
previous_position = None

# Initialize the scale factor for translation
scale = 1

# Create lists to store the positions for visualization
positions_x = [current_position[0, 0]]
positions_y = [current_position[1, 0]]
positions_z = [current_position[2, 0]]

# Set the URL for the IP camera
url = 'https://192.168.129.235:8080/video'

# Set up the camera object
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Error opening video stream or file")

# Create a window for displaying the live camera feed
cv2.namedWindow('Live Camera Feed', cv2.WINDOW_NORMAL)

# Create a 3D plot for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

orb = cv2.ORB_create()
old_keypoints = None
old_descriptors = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame is not None:
        # Perform feature detection on the current frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

        if previous_position is not None and old_descriptors is not None:
            # Use feature matching to track the features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(old_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract the matching keypoints
            src_pts = np.float32([old_keypoints[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches])

            # Calculate the essential matrix
            essential_matrix, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=0.5)

            # Extract the camera motion (rotation and translation) from the essential matrix
            _, R, t, _ = cv2.recoverPose(essential_matrix, src_pts, dst_pts)

            # Update the current position based on the translation
            current_position += scale * R.dot(t)

            # Print the current position
            print("Current Position:")
            print(current_position)

            # Store the positions for visualization
            positions_x.append(current_position[0, 0])
            positions_y.append(current_position[1, 0])
            positions_z.append(current_position[2, 0])

            # Clear the previous plot and update the 3D plot
            ax.clear()
            ax.plot(positions_x, positions_y, positions_z)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.pause(0.001)

        # Update the previous position and features
        previous_position = current_position.copy()
        old_keypoints = keypoints
        old_descriptors = descriptors

        # Display the processed frame with detected features
        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
        cv2.imshow('Live Camera Feed', frame_with_keypoints)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
