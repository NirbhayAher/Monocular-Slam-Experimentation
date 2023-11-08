import cv2
import torch
from IPython.display import clear_output
import numpy as np

# Load the MiDaS model
model_type = "DPT_Large"  # MiDaS v2.1 - Small (lowest accuracy, highest inference speed)
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

        # Normalize the depth map for visualization
        prediction = prediction / prediction.max() * 255
        prediction = prediction.astype(np.uint8)

        # Display the live depth map
        cv2.imshow('Live Depth Map', prediction)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        clear_output(wait=True)

except KeyboardInterrupt:
    cap.release()

# Release the camera
cap.release()
cv2.destroyAllWindows()
