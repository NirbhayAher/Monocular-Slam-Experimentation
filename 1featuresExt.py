import cv2

# Set up the camera object
cap = cv2.VideoCapture(0)

try:
    # Create a window for displaying the live camera feed
    cv2.namedWindow('Live Camera Feed', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        
        # Perform feature detection and matching on the current frame
        orb = cv2.ORB_create()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_frame, None)
        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)

        # Display the processed frame with detected features
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
