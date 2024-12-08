import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from matplotlib.colors import hsv_to_rgb
from InceptionNext import InceptionNext

# Klasa datasetu
from autoflowdataset import AutoFlowAug2Dataset

def visualize_flow(flow):
    """
    Converts optical flow to a visual representation using HSV color space.

    Args:
        flow (torch.Tensor): Optical flow tensor (H, W, 2).

    Returns:
        np.ndarray: RGB image representing the optical flow.
    """
    flow = flow.permute(1, 2, 0).numpy()  # H, W, 2
    magnitude = np.linalg.norm(flow, axis=2)
    angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])

    # Normalize magnitude to [0, 1]
    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude) + 1e-5)
    
    # Convert to HSV
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi)  # Hue (0 to 1 based on angle)
    hsv[..., 1] = 1.0  # Full saturation
    hsv[..., 2] = magnitude  # Value based on magnitude

    rgb = hsv_to_rgb(hsv)  # Convert HSV to RGB
    return rgb

def visualize_dataset(dataset_path, sample_count=5):
    """
    Visualizes the dataset.

    Args:
        dataset_path (str): Path to the dataset root directory.
        sample_count (int): Number of samples to visualize.
    """
    dataset = AutoFlowAug2Dataset(root_dir=dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, ((im0, im1), target) in enumerate(dataloader):
        if i >= sample_count:
            break

        # Prepare images
        im0 = to_pil_image(im0[0])
        im1 = to_pil_image(im1[0])

        # Prepare optical flow visualizations
        target = target[0]  # Remove batch dimension
        flows = [visualize_flow(target[j:j+2]) for j in range(0, 8, 2)]

        # Plot the images and optical flow
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))

        # Input images
        axes[0, 0].imshow(im0)
        axes[0, 0].set_title("im0")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(im1)
        axes[0, 1].set_title("im1")
        axes[0, 1].axis("off")

        # Optical flow visualizations
        for idx, flow in enumerate(flows):
            ax = axes[1, idx]
            ax.imshow(flow)
            ax.set_title(f"Flow {idx+1}")
            ax.axis("off")

        # Hide unused axes
        for j in range(len(flows), 5):
            axes[1, j].axis("off")

        plt.tight_layout()
        plt.show()


def test_model(model, dataset_path, sample_count=5, device="cpu"):
    """
    Tests a trained model on the dataset and visualizes the predictions.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataset_path (str): Path to the dataset root directory.
        sample_count (int): Number of samples to test.
        device (str): Device to run the model on ("cpu" or "cuda").
    """
    model.to(device)
    model.eval()

    dataset = AutoFlowAug2Dataset(root_dir=dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, ((im0, im1), target) in enumerate(dataloader):
        if i >= sample_count:
            break

        # Prepare input
        im0, im1 = im0.to(device), im1.to(device)
        inputs = (im0, im1)

        # Model prediction
        with torch.no_grad():
            prediction = model(*inputs)

        # Prepare images
        im0 = to_pil_image(im0[0].cpu())
        im1 = to_pil_image(im1[0].cpu())

        # Prepare optical flow visualizations
        target = target[0]  # Remove batch dimension
        flows_target = [visualize_flow(target[j:j+2]) for j in range(0, 8, 2)]

        prediction = prediction[0].cpu()  # Remove batch dimension and move to CPU
        flows_pred = [visualize_flow(prediction[j:j+2]) for j in range(0, 8, 2)]

        # Plot the images, ground truth and predictions
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))

        # Input images
        axes[0, 0].imshow(im0)
        axes[0, 0].set_title("im0")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(im1)
        axes[0, 1].set_title("im1")
        axes[0, 1].axis("off")

        # Ground truth optical flow
        for idx, flow in enumerate(flows_target):
            ax = axes[1, idx]
            ax.imshow(flow)
            ax.set_title(f"GT Flow {idx+1}")
            ax.axis("off")

        # Predicted optical flow
        for idx, flow in enumerate(flows_pred):
            ax = axes[2, idx]
            ax.imshow(flow)
            ax.set_title(f"Pred Flow {idx+1}")
            ax.axis("off")

        # Hide unused axes
        for j in range(len(flows_target), 5):
            axes[1, j].axis("off")
            axes[2, j].axis("off")

        plt.tight_layout()
        plt.show()


def test_external_images(model, image1_path, image2_path, device="cpu"):
    """
    Tests a trained model on external images and visualizes the predictions.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        image1_path (str): Path to the first image (im0).
        image2_path (str): Path to the second image (im1).
        device (str): Device to run the model on ("cpu" or "cuda").
    """
    from torchvision.io import read_image

    model.to(device)
    model.eval()

    # Load and prepare images
    im0 = read_image(image1_path).float() / 255.0  # Normalize to [0, 1]
    im1 = read_image(image2_path).float() / 255.0

    im0, im1 = im0.unsqueeze(0).to(device), im1.unsqueeze(0).to(device)  # Add batch dimension

    # Model prediction
    with torch.no_grad():
        prediction = model(im0, im1)

    # Convert images for visualization
    im0 = to_pil_image(im0[0].cpu())
    im1 = to_pil_image(im1[0].cpu())

    # Prepare optical flow visualizations
    prediction = prediction[0].cpu()  # Remove batch dimension and move to CPU
    flows_pred = [visualize_flow(prediction[j:j+2]) for j in range(0, 8, 2)]

    # Plot the images and predictions
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Input images
    axes[0, 0].imshow(im0)
    axes[0, 0].set_title("im0")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(im1)
    axes[0, 1].set_title("im1")
    axes[0, 1].axis("off")

    # Predicted optical flow
    for idx, flow in enumerate(flows_pred):
        ax = axes[1, idx]
        ax.imshow(flow)
        ax.set_title(f"Pred Flow {idx+1}")
        ax.axis("off")

    # Hide unused axes
    for j in range(len(flows_pred), 5):
        axes[1, j].axis("off")

    plt.tight_layout()
    plt.show()

# Użycie funkcji

#visualize_dataset("/mnt/d/datasety/autoflowaug2", sample_count=5)

model = InceptionNext()
checkpoint = torch.load("inceptionnext3.pth", weights_only=True)
model.load_state_dict(checkpoint)

test_model(model, "/mnt/d/datasety/autoflowaug2", sample_count=5)
#test_external_images(model, "/mnt/d/flow2.jpg", "/mnt/d/flow1.jpg", device="cpu")
