import cv2
import numpy as np

# Initialize the initial position as the origin
current_position = np.zeros((3, 1))

# Initialize the previous position
previous_position = np.zeros((3, 1))

# Initialize the scale factor for translation
scale = 1

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
    delay_duration = 10

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

            # Extract the matching keypoints
            src_pts = np.float32([old_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Calculate the essential matrix
            essential_matrix, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=0.5)

            # Extract the camera motion (rotation and translation) from the essential matrix
            _, R, t, _ = cv2.recoverPose(essential_matrix, src_pts, dst_pts)

            # Update the current position based on the translation
            current_position += scale * R.dot(t)

            # Print the current position
            print("Current Position:")
            print(current_position)

            # Update the previous position
            previous_position = current_position

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
