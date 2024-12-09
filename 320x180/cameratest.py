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
    # Resize frames to ensure they are consistent with model input size (640x360 in this case)
    frame_resized = cv2.resize(frame, (640, 360))
    prev_frame_resized = cv2.resize(prev_frame, (640, 360))

    # Convert to tensor and normalize to [0, 1]
    current_tensor = to_tensor(frame_resized).unsqueeze(0).to(device)
    prev_tensor = to_tensor(prev_frame_resized).unsqueeze(0).to(device)

    # Pass through model
    with torch.no_grad():
        output = model(prev_tensor, current_tensor)

    # Extract first optical flow tensor (shape: 2x18x32)
    optical_flow = output[0, :2, :, :].cpu().numpy()

    return frame_resized, prev_frame_resized, optical_flow


def draw_optical_flow(frame, flow, scale=1):
    h, w = frame.shape[:2]
    step = 20

    # Create a grid of points
    y, x = np.mgrid[step // 2:h:step, step // 2:w:step]

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


def draw_optical_flow_color(frame, flow):
    h, w = frame.shape[:2]
    flow = cv2.resize(flow.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
    # Rozdziel składowe przepływu optycznego
    flow_x, flow_y = flow[..., 0], flow[..., 1]

    # Oblicz wielkość (magnitude) i kąt (angle)
    mag, ang = cv2.cartToPolar(flow_x, flow_y)

    # Znormalizuj wielkość do zakresu [0, 255]
    #mag_normalized = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Stwórz obraz HSV
    hsv_image = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_image[..., 0] = ang * (180 / np.pi / 2)  # Przelicz kąt na stopnie dla zakresu Hue [0, 179]
    hsv_image[..., 1] = 255                      # Stała saturacja
    hsv_image[..., 2] = 2*mag.astype(np.uint8)  # Jasność bazująca na wielkości

    # Konwertuj HSV na RGB
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return rgb_image


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
        _, prev_frame_resized, optical_flow = process_frame(frame, prev_frame)

        # Draw optical flow vectors on the frame
        result_frame = draw_optical_flow(frame.copy(), optical_flow)

        # Draw optical flow in color for visualization
        color_optical_flow = draw_optical_flow_color(prev_frame_resized, optical_flow)  # Use resized previous frame here

        cv2.imshow('Optical Flow', result_frame)
        cv2.imshow('Color Optical Flow', color_optical_flow)  # New window for color visualization

        # Update previous frame and break if 'q' is pressed
        prev_frame = frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end_time = time.time()
        #print(f"Frame processing time: {end_time - start_time:.3f}s")
finally:
    cap.release()
    cv2.destroyAllWindows()
