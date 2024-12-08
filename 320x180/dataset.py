import os
from torch.utils.data import Dataset
import torch
import torchvision.io as io
import torchvision.transforms as T
import numpy as np
import random
import json

class AutoFlowAug2Dataset(Dataset):
    def __init__(self, root_dir, transform=None, cache_file="dataset_cache.json"):
        """
        Args:
            root_dir (str): Ścieżka do głównego folderu datasetu 'autoflowaug2'.
            transform (callable, optional): Transformacje do zastosowania na obrazach.
            cache_file (str): Ścieżka do pliku cache z listą folderów.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.cache_file = os.path.join(root_dir, cache_file)

        if os.path.exists(self.cache_file):
            # Wczytaj listę folderów z cache
            with open(self.cache_file, 'r') as f:
                self.folders = json.load(f)
        else:
            # Stwórz listę folderów i zapisz do pliku cache
            self.folders = [
                folder_name for folder_name in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, folder_name))
            ]
            with open(self.cache_file, 'w') as f:
                json.dump(self.folders, f)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder_name = self.folders[idx]
        folder_path = os.path.join(self.root_dir, folder_name)

        im0_path = os.path.join(folder_path, "im0.jpg")
        im1_path = os.path.join(folder_path, "im1.jpg")
        target_path = os.path.join(folder_path, "max_median_min_mean.npy")

        # Wczytanie obrazów za pomocą torchvision.io
        im0 = io.read_image(im0_path).float() / 255.0
        im1 = io.read_image(im1_path).float() / 255.0

        # Wczytanie pliku numpy
        target = np.load(target_path).astype(np.float32)  # Konwersja na float32
        target = torch.tensor(target)

        # Zastosowanie augmentacji
        if random.random() < 0.2:
            # 1. Desaturacja
            alpha0 = random.random()
            alpha1_min = max(0, alpha0 - 0.4)
            alpha1_max = min(1, alpha0 + 0.4)
            alpha1 = random.uniform(alpha1_min, alpha1_max)
            
            im0_avg = torch.mean(im0, dim=0).unsqueeze(0).repeat(3, 1, 1)
            im0 = alpha0 * im0 + (1 - alpha0) * im0_avg
            im1_avg = torch.mean(im1, dim=0).unsqueeze(0).repeat(3, 1, 1)
            im1 = alpha1 * im1 + (1 - alpha1) * im1_avg
        
        if random.random() < 0.3:
            # 1. Kontrast
            r = random.random()*0.5+0.75
            im0 = torch.clamp((im0-0.5)*r+0.5, 0, 1)
            im1 = torch.clamp((im1-0.5)*r+0.5, 0, 1)
            im1 = torch.clamp((im1-0.5)*(random.random()*0.4+0.8)+0.5, 0, 1)
        
        if random.random() < 0.3:
            # 2. Jasność
            r = (random.random()-0.5)*0.4
            im0 = torch.clamp(im0 + r, 0, 1)
            im1 = torch.clamp(im1 + r, 0, 1)
            im1 = torch.clamp(im1 + (random.random()-0.5)*0.2, 0, 1)
        
        if random.random() < 0.5:
            # 3. Odwrócenie poziome
            im0 = T.functional.hflip(im0)
            im1 = T.functional.hflip(im1)
            target = target.flip(-1)  # Odwrócenie w osi poziomej
            target[::2] = -target[::2] # Odwrócenie wektorów w osi poziomej

        if random.random() < 0.1:
            # 4. Odwrócenie pionowe
            im0 = T.functional.vflip(im0)
            im1 = T.functional.vflip(im1)
            target = target.flip(-2)  # Odwrócenie w osi pionowej
            target[1::2] = -target[1::2] # Odwrócenie wektorów w osi pionowej

        if random.random() < 0.2:
            # 5. Rozmycie (blur)
            blur = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.2, 2.0))
            im0 = blur(im0)
            im1 = blur(im1)

        if random.random() < 0.2:
            # 6. Wyostrzanie
            sharpen = T.Compose([
                T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
                T.functional.adjust_sharpness
            ])
            im0 = T.functional.adjust_sharpness(im0, sharpness_factor=4)
            im1 = T.functional.adjust_sharpness(im1, sharpness_factor=4)

        if random.random() < 0.3:
            # 7. Dodanie szumu
            r = random.random()
            noise = torch.randn_like(im0) * 0.2 * r
            im0 = torch.clamp(im0 + noise, 0, 1)
            noise = torch.randn_like(im0) * 0.2 * r
            im1 = torch.clamp(im1 + noise, 0, 1)
        
        # Zastosowanie transformacji (jeśli podane)
        if self.transform:
            im0 = self.transform(im0)
            im1 = self.transform(im1)

        return (im0, im1), target
      
