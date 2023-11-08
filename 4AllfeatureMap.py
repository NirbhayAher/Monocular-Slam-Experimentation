import cv2
import numpy as np
import time

# Set up the camera object
cap = cv2.VideoCapture(0)

# Initialize the feature map
feature_map = np.zeros((480, 640, 3), np.uint8)

try:
    # Create a window for displaying the live camera feed
    cv2.namedWindow('Live Camera Feed', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()

        # Perform feature detection on the current frame
        orb = cv2.ORB_create()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

        # Draw the tracked features on the feature map
        for point in keypoints:
            x, y = map(int, point.pt)
            cv2.circle(feature_map, (x, y), 5, (0, 0, 255), -1)

        # Display the feature map
        cv2.imshow('Feature Map', feature_map)

        # Display the live camera feed with detected features
        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
        cv2.imshow('Live Camera Feed', frame_with_keypoints)

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
