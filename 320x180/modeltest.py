import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from InceptionNext import InceptionNext


def load_image_as_tensor(filepath):
    """
    Wczytuje obraz z pliku i przekształca go na tensor RGB o wymiarach 3x180x320 (CHW).

    :param filepath: Ścieżka do pliku obrazu
    :return: Tensor obrazu RGB (3x180x320)
    """
    image = Image.open(filepath).convert('RGB')
    image = image.resize((320, 180))
    image_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
    return image_tensor

def test_and_visualize_model(model, image1, image2):
    """
    Testuje model na dwóch obrazach i wyświetla pole wektorów pierwszego tensora optical flow.

    :param model: PyTorch model, który przyjmuje dwa tensory obrazów i zwraca tensor optical flow
    :param image1: Tensor obrazu RGB (3x180x320)
    :param image2: Tensor obrazu RGB (3x180x320)
    """
    # Upewnij się, że obrazy są w odpowiednim formacie
    if image1.shape != (3, 180, 320) or image2.shape != (3, 180, 320):
        raise ValueError("Obrazy muszą mieć wymiary 3x180x320 (CHW).")

    # Przełącz model w tryb ewaluacji
    model.eval()

    # Przetestuj model
    with torch.no_grad():
        output = model(image1.unsqueeze(0), image2.unsqueeze(0))

    # Output ma wymiary 8x18x32, podzielony na cztery tensory optical flow po 2x18x32
    if output.shape != (1, 8, 18, 32):
        raise ValueError("Wyjście modelu powinno mieć wymiary 8x18x32 (CHW).")

    # Wybierz pierwszy tensor optical flow (2x18x32)
    optical_flow = output[0, :2, :, :].cpu().numpy()

    # Przygotuj dane do wizualizacji
    u = optical_flow[0]  # Składowa pozioma
    v = optical_flow[1]  # Składowa pionowa

    # Stwórz siatkę punktów odpowiadającą optycznemu przepływowi
    x = np.linspace(0, 320, 32)
    y = np.linspace(0, 180, 18)
    xv, yv = np.meshgrid(x, y)

    # Przekształć obraz na format HWC i na numpy
    image1_np = image1.permute(1, 2, 0).cpu().numpy()
    image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min())  # Normalizacja do [0, 1]

    # Wyświetl obraz z polem wektorów
    plt.figure(figsize=(12, 6))
    plt.imshow(image1_np)
    plt.quiver(xv, yv, u, v, color='r', angles='xy', scale_units='xy', scale=0.2, width=0.0025)
    plt.title("Optical Flow na Obrazie")
    plt.axis('off')
    plt.show()


model = InceptionNext()
checkpoint = torch.load("inceptionnext2.pth", weights_only=True)
model.load_state_dict(checkpoint)

image_path1 = "/mnt/d/flow1.jpg"
image_path2 = "/mnt/d/flow2.jpg"
image1 = load_image_as_tensor(image_path1)
image2 = load_image_as_tensor(image_path2)
test_and_visualize_model(model, image1, image2)
