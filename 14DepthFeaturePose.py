import cv2
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

# Initialize the initial position as the origin
current_position = np.zeros((3, 1), dtype=np.float32)

# Initialize the scale factor for translation
scale = 1

# Create lists to store the positions for visualization
positions_x = [current_position[0, 0]]
positions_y = [current_position[2, 0]]  # Adjust the positions_y to reflect the z-axis
positions_z = [current_position[1, 0]]  # Adjust the positions_z to reflect the y-axis

# Set up the camera object
cap = cv2.VideoCapture(0)

# Create a 3D plot for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Z Label')  # Adjust the y-axis label to reflect the z-axis
ax.set_zlabel('Y Label')  # Adjust the z-axis label to reflect the y-axis

# Initialize the variables for feature tracking
old_frame = None
old_keypoints = None
old_descriptors = None

# Set the delay duration (in seconds)
delay_duration = 0.1

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

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
