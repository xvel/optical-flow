import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor, resize
from InceptionNext import InceptionNext  # Ensure this is the correct path to your model class file
import time

# Load the model and move it to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionNext().to(device)
checkpoint = torch.load("inceptionnext4.pth", map_location=device)  # Ensure this path is correct and the file exists
model.load_state_dict(checkpoint)


def process_frame(frame, prev_frame):
    # Resize and normalize frames
    frame_resized = cv2.resize(frame, (320, 180))
    prev_frame_resized = cv2.resize(prev_frame, (320, 180))

    # Convert to tensor and normalize to [0, 1]
    current_tensor = to_tensor(frame_resized).unsqueeze(0).to(device)
    prev_tensor = to_tensor(prev_frame_resized).unsqueeze(0).to(device)

    # Pass through model
    with torch.no_grad():
        output = model(prev_tensor, current_tensor)

    # Extract first optical flow tensor (shape: 2x18x32)
    optical_flow = output[0, :2, :, :].cpu().numpy()

    return optical_flow


def draw_optical_flow(frame, flow, scale=1):
    h, w = frame.shape[:2]
    step = 40

    # Create a grid of points
    y, x = np.mgrid[step//2:h:step, step//2:w:step]

    # Scale flow to match frame size
    flow = cv2.resize(flow.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
    u, v = flow[..., 0], flow[..., 1]

    # Draw vectors on the frame
    for i in range(0, y.shape[0]):
        for j in range(0, x.shape[1]):
            start_point = (x[i, j].item(), y[i, j].item())  # Convert to Python scalar type
            end_point = (int(x[i, j] + scale * u[y[i, j], x[i, j]]), int(y[i, j] + scale * v[y[i, j], x[i, j]]))
            cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 1, tipLength=0.3)

    return frame


# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)
if not cap.isOpened():
    print("Nie udało się otworzyć kamery.")
    exit()

ret, prev_frame = cap.read()
if not ret:
    print("Nie udało się pobrać pierwszej klatki.")
    cap.release()
    exit()

try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break

        # Process frames and move them to GPU if available
        optical_flow = process_frame(frame, prev_frame)

        # Draw optical flow vectors on the frame
        result_frame = draw_optical_flow(frame.copy(), optical_flow)

        cv2.imshow('Optical Flow', result_frame)

        # Update previous frame and break if 'q' is pressed
        prev_frame = frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end_time = time.time()
        print(f"Frame processing time: {end_time - start_time:.3f}s")
finally:
    cap.release()
    cv2.destroyAllWindows()
