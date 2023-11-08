import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize the initial position as the origin
current_position = np.zeros((3, 1), dtype=np.float32)

# Initialize the previous position
previous_position = None

# Initialize the scale factor for translation
scale = 1

# Create lists to store the positions for visualization
positions_x = [current_position[0, 0]]
positions_y = [current_position[2, 0]]  # Adjust the positions_y to reflect the z-axis
positions_z = [current_position[1, 0]]  # Adjust the positions_z to reflect the y-axis

# Set up the camera object
cap = cv2.VideoCapture(0)

# Create a window for displaying the live camera feed
cv2.namedWindow('Live Camera Feed', cv2.WINDOW_NORMAL)

# Create a 3D plot for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Z Label')  # Adjust the y-axis label to reflect the z-axis
ax.set_zlabel('Y Label')  # Adjust the z-axis label to reflect the y-axis

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
            matches = bf.match(old_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract the reliable matching keypoints
            good_matches = [m for m in matches if m.distance < 50]  # Adjust the threshold for reliable features

            src_pts = np.float32([old_keypoints[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

            # Calculate the essential matrix using RANSAC
            essential_matrix, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=0.5)

            if mask is not None and np.sum(mask) > 15:
                src_pts = src_pts[mask.ravel() == 1]
                dst_pts = dst_pts[mask.ravel() == 1]

                essential_matrix, _ = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=0.5)

                _, R, t, _ = cv2.recoverPose(essential_matrix, src_pts, dst_pts)

                current_position += scale * R.dot(t)

                print("Current Position:")
                print(current_position)

                positions_x.append(current_position[0, 0])
                positions_y.append(current_position[2, 0])  # Adjust the positions_y to reflect the z-axis
                positions_z.append(current_position[1, 0])  # Adjust the positions_z to reflect the y-axis

                ax.clear()
                ax.plot(positions_x, positions_y, positions_z)
                ax.set_xlabel('X Label')
                ax.set_ylabel('Z Label')  # Adjust the y-axis label to reflect the z-axis
                ax.set_zlabel('Y Label')  # Adjust the z-axis label to reflect the y-axis
                plt.pause(0.001)

        previous_position = current_position.copy()
        old_keypoints = keypoints
        old_descriptors = descriptors

        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
        cv2.imshow('Live Camera Feed', frame_with_keypoints)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
