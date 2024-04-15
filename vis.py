import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from autoflowaug import AutoFlowAugDataset  # Załóżmy, że zapisaliśmy klasę datasetu w pliku o nazwie 'dataset.py'
import cv2

def visualize_optical_flow(flow):
    # Wizualizuj optical flow za pomocą kolorowego obrazu
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Konwertuj HSV do RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb

# Tworzenie instancji datasetu
dataset = AutoFlowAugDataset(root_dir="/mnt/d/datasety/autoflowaug")

# Wybór przykładu do wizualizacji
example_idx = 123+9*40000
im0, im1, flow = dataset[example_idx]

# Wizualizacja
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Wyświetlenie pierwszego i drugiego obrazu
axes[0].imshow(np.transpose(im0.numpy(), (1, 2, 0)))
axes[0].set_title("Image 0")
axes[1].imshow(np.transpose(im1.numpy(), (1, 2, 0)))
axes[1].set_title("Image 1")

# Wyświetlenie optical flow
optical_flow_img = visualize_optical_flow(flow.permute(1, 2, 0).numpy())
axes[2].imshow(optical_flow_img)
axes[2].set_title("Optical Flow")

plt.show()
