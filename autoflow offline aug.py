import os
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.io import read_image
from torch.utils.data import DataLoader
from autoflow import AutoFlowDataset
import numpy as np

def save_transformed_dataset(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, (img0, img1, flo_data) in enumerate(dataset):
        sample_dir = os.path.join(save_dir, f'sample_{i}')
        os.makedirs(sample_dir, exist_ok=True)

        # Zapisywanie obrazów jako JPG z jakością 90
        img0_path = os.path.join(sample_dir, 'im0.jpg')
        img1_path = os.path.join(sample_dir, 'im1.jpg')
        img0_pil = to_pil_image(img0)
        img1_pil = to_pil_image(img1)
        img0_pil.save(img0_path, 'JPEG', quality=95)
        img1_pil.save(img1_path, 'JPEG', quality=95)

        # Zapisywanie danych optical flow jako plik .npy
        flo_path = os.path.join(sample_dir, 'forward.npy')
        np.save(flo_path, flo_data.numpy().astype(np.float16))

root_dir = ''
save_dir = ''

dataset = AutoFlowDataset(root_dir)

save_transformed_dataset(dataset, save_dir)
