import cv2
import torch
import numpy as np
from IPython.display import clear_output

# Load the MiDaS model
model_type = "MiDaS_small"  # MiDaS v2.1 - Small (lowest accuracy, highest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Initialize the variables for feature tracking
old_frame = None
old_keypoints = None
old_descriptors = None

# Set the delay duration (in seconds)
delay_duration = 0.1

try:
    while True:
        ret, frame = cap.read()

        # Perform feature detection on the current frame
        orb = cv2.ORB_create()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

        # Overlay the tracked features onto the depth map
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        # Normalize the depth map for visualization
        prediction = prediction / prediction.max() * 255
        prediction = prediction.astype(np.uint8)

        # Draw the keypoints on the depth map
        for keypoint in keypoints:
            x, y = map(int, keypoint.pt)
            cv2.circle(prediction, (x, y), 5, (0, 255, 0), -1)

        # Display the live depth map with overlaid features
        cv2.imshow('Live Depth Map with Tracked Features', prediction)

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

        clear_output(wait=True)

except Exception as e:
    # Clean up the camera object in case of errors
    print('An error occurred:', e)
    cap.release()
    cv2.destroyAllWindows()

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
