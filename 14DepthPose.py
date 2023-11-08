import cv2
import torch
import numpy as np
import open3d as o3d
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Load the MiDaS model
model_type = "MiDaS_small"
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

try:
    while True:
        ret, frame = cap.read()

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

        # Normalize the depth map and convert it to meters
        prediction = prediction / prediction.max()
        depth_scale = 1.0
        prediction *= depth_scale

        # Capture the camera pose
        if frame is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

            if previous_position is not None and old_descriptors is not None:
                matches = bf.match(old_descriptors, descriptors)
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = [m for m in matches if m.distance < 50]

                src_pts = np.float32([old_keypoints[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

                essential_matrix, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=0.5)

                if mask is not None and np.sum(mask) > 15:
                    src_pts = src_pts[mask.ravel() == 1]
                    dst_pts = dst_pts[mask.ravel() == 1]

                    essential_matrix, _ = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=0.5)

                    _, R, t, _ = cv2.recoverPose(essential_matrix, src_pts, dst_pts)

                    current_position += scale * R.dot(t)

                    positions_x.append(current_position[0, 0])
                    positions_y.append(current_position[2, 0])
                    positions_z.append(current_position[1, 0])

                    ax.clear()
                    ax.plot(positions_x, positions_y, positions_z)
                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Z Label')
                    ax.set_zlabel('Y Label')
                    plt.pause(0.001)

            previous_position = current_position.copy()
            old_keypoints = keypoints
            old_descriptors = descriptors

        # Create a mask to exclude invalid depth values
        mask = prediction > 0

        # Generate the point cloud from the depth map
        pixel_coords = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        pixel_coords = np.column_stack((pixel_coords[1].ravel(), pixel_coords[0].ravel()))
        depth_values = prediction.ravel()
        valid_points = np.all(np.isfinite(depth_values) & mask.ravel())

        depth_values = depth_values[valid_points]
        pixel_coords = pixel_coords[valid_points]

        intrinsic_matrix = np.eye(3)
        
        points = depth_values.reshape(-1, 1) * np.linalg.inv(intrinsic_matrix).dot(
            np.column_stack((pixel_coords, np.ones((pixel_coords.shape[0], 1)))
        ).T
        )

        # Transform points to world coordinates using the camera pose
        points = np.dot(extrinsic_matrix[:3, :3], points.T) + extrinsic_matrix[:3, 3][:, np.newaxis]

        # Set point cloud points
        point_cloud.points = o3d.utility.Vector3dVector(points.T)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud])

        # Display the live depth map
        cv2.imshow('Live Depth Map', (prediction * 255).astype(np.uint8))

        # Display the live camera feed
        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
        cv2.imshow('Live Camera Feed', frame_with_keypoints)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        clear_output(wait=True)

except KeyboardInterrupt:
    cap.release()

# Release the camera
cap.release()
cv2.destroyAllWindows()
