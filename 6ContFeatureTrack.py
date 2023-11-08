import cv2
import numpy as np

# Set up the camera object
cap = cv2.VideoCapture(0)

# Initialize the feature map
feature_map = np.zeros((480, 640, 3), np.uint8)

# 3D coordinates of the tracked points (assuming they lie on the same plane)
world_points = np.array([[0, 0, 0],
                         [100, 0, 0],
                         [100, 100, 0],
                         [0, 100, 0]], dtype=np.float32)

# Initialize the 3D points and 2D points for the PnP algorithm
obj_points = []
img_points = []

try:
    while True:
        ret, frame = cap.read()

        # Perform feature detection on the current frame
        orb = cv2.ORB_create()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

        # Clear the feature map before updating it with new positions
        feature_map = np.zeros((480, 640, 3), np.uint8)

        # Draw the tracked features on the feature map
        for point in keypoints:
            x, y = map(int, point.pt)
            cv2.circle(feature_map, (x, y), 5, (0, 0, 255), -1)

        # Display the feature map
        cv2.imshow('Feature Map', feature_map)

        # Display the live camera feed with detected features
        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
        cv2.imshow('Live Camera Feed', frame_with_keypoints)

        # Add 3D-2D point correspondences for the PnP algorithm
        if len(keypoints) >= 4:  # Use at least 4 points for the PnP algorithm
            obj_points = []
            img_points = []
            for i in range(4):  # Adjust the range according to the number of points in world_points
                obj_points.append(world_points[i])
                img_points.append(keypoints[i].pt)

            obj_points = np.array(obj_points, dtype=np.float32)
            img_points = np.array(img_points, dtype=np.float32)

            # Find the homography and remove outliers using RANSAC
            _, mask = cv2.findHomography(obj_points, img_points, cv2.RANSAC, 5.0)
            inliers = mask.ravel().tolist()

            obj_points_filtered = [obj_points[i] for i in range(len(obj_points)) if inliers[i] == 1]
            img_points_filtered = [img_points[i] for i in range(len(img_points)) if inliers[i] == 1]

            # Estimate the camera pose using the PnP algorithm
            _, rvec, tvec = cv2.solvePnP(np.array(obj_points_filtered), np.array(img_points_filtered), np.eye(3), np.zeros(5))

            # Apply the transformation matrix to the detected features
            for point in keypoints:
                point_3d = np.array([[point.pt[0], point.pt[1], 0]], dtype=np.float32)
                point_3d = np.array([point_3d])
                point_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, np.eye(3), np.zeros(5))
                point.pt = (point_2d[0][0][0], point_2d[0][0][1])

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    # Clean up the camera object in case of errors
    print('An error occurred:', e)
    cap.release()
    cv2.destroyAllWindows()

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
