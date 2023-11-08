import cv2
import numpy as np
import time

# Set up the camera object
cap = cv2.VideoCapture(0)

try:
    # Create a window for displaying the live camera feed
    cv2.namedWindow('Live Camera Feed', cv2.WINDOW_NORMAL)

    # Initialize the variables for feature tracking
    old_frame = None
    old_keypoints = None
    old_descriptors = None

    # Set the delay duration (in seconds)
    delay_duration = 1

    while True:
        ret, frame = cap.read()

        # Perform feature detection on the current frame
        orb = cv2.ORB_create()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

        if old_frame is not None and old_keypoints is not None and old_descriptors is not None:
            # Use feature matching to track the features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(old_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            # Draw the matches
            matched_frame = cv2.drawMatches(old_frame, old_keypoints, frame, keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Display the processed frame with tracked features
            cv2.imshow('Feature Tracking', matched_frame)
            cv2.waitKey(int(delay_duration * 1000))  # Delay the display for a certain duration

        # Update the previous frame and its features
        old_frame = frame.copy()
        old_keypoints = keypoints
        old_descriptors = descriptors

        # Display the processed frame with detected features
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

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
