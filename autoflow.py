import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os


class AutoFlowDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = os.listdir(root_dir)

    @staticmethod
    def read_flo_file(file_path):
        with open(file_path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if magic != 202021.25:
                raise Exception('Nieprawidłowy format .flo')
            # Odczytanie szerokości i wysokości
            width = np.fromfile(f, np.int32, count=1)[0]
            height = np.fromfile(f, np.int32, count=1)[0]
            # Odczytanie danych optical flow i reshape
            data = np.fromfile(f, np.float32, count=2 * width * height)
            return torch.from_numpy(data.reshape((height, width, 2)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path = os.path.join(self.root_dir, self.samples[idx])
        # Ścieżki do obrazów i pliku .flo
        img0_path = os.path.join(folder_path, 'im0.png')
        img1_path = os.path.join(folder_path, 'im1.png')
        flo_path = os.path.join(folder_path, 'forward.flo')

        # Ładowanie i konwersja obrazów
        img0 = read_image(img0_path).float()/255
        img1 = read_image(img1_path).float()/255

        # Odczytanie i konwersja danych optical flow do tensora
        flo_data = self.read_flo_file(flo_path).permute(2, 0, 1)

        # Losowe przycięcie na rozmiar 320x180 lub skalowanie na 324x252 i przycięcie na 320x180
        h, w = img0.shape[1], img0.shape[2]
        if random.random() < 0.3:
            # Skalowanie na 324x252 i przycięcie na 320x180
            new_h, new_w = 252, 324
            img0_rescaled = F.interpolate(img0.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
            img1_rescaled = F.interpolate(img1.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
            flo_data_rescaled = F.interpolate(flo_data.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
            top = random.randint(0, new_h - 180)
            left = random.randint(0, new_w - 320)
            img0 = img0_rescaled[:, top: top + 180, left: left + 320]
            img1 = img1_rescaled[:, top: top + 180, left: left + 320]
            flo_data = flo_data_rescaled[:, top: top + 180, left: left + 320]
            flo_data *= torch.tensor([0.5625, 0.5625], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        else:
            # Random crop na rozmiar 320x180
            new_h, new_w = 180, 320
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            img0 = img0[:, top: top + new_h, left: left + new_w]
            img1 = img1[:, top: top + new_h, left: left + new_w]
            flo_data = flo_data[:, top: top + new_h, left: left + new_w]

        # Losowe odwrócenie poziome
        if random.random() < 0.5:
            img0 = torch.flip(img0, [2])
            img1 = torch.flip(img1, [2])
            flo_data = torch.flip(flo_data, [2])
            flo_data[0] *= -1  # Odwrócenie wektora x
        # Losowe odwrócenie pionowe
        if random.random() < 0.5:
            img0 = torch.flip(img0, [1])
            img1 = torch.flip(img1, [1])
            flo_data = torch.flip(flo_data, [1])
            flo_data[1] *= -1  # Odwrócenie wektora y

        # losowa zmiana ekspozycji
        if random.random() < 0.5:
            img0 = (img0 * (((torch.rand(1)-0.5)*0.5)+1.0)).clamp(max=1.0)

        # losowa zmiana jasności
        if random.random() < 0.5:
            img0 = (img0 + ((torch.rand(1)-0.5)*0.5)).clamp(min=0.0, max=1.0)

        # losowa zmiana kolorów
        if random.random() < 0.5:
            img1 = (img1 * (((torch.rand(3, 1, 1)-0.5)*0.1)+1.0)).clamp(max=1.0)

        # Losowe odwrócenie kolorów
        if random.random() < 0.2:
            img0 = 1.0-img0
            img1 = 1.0-img1

        # Losowy szum
        if random.random() < 0.5:
            s = torch.rand(1)*0.5
            img0 = (img0 + ((torch.rand(3, 180, 320)-0.5)*s)).clamp(min=0.0, max=1.0)
            img1 = (img1 + ((torch.rand(3, 180, 320)-0.5)*s)).clamp(min=0.0, max=1.0)
        
        return img0, img1, flo_data
