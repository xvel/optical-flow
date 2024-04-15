import numpy as np
import cv2
import matplotlib.pyplot as plt

def flow_to_color(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def read_flow_file(file):
    try:
        flow = np.load(file)
        flow = flow.astype(np.float32)
        return np.transpose(flow, (1, 2, 0))
    except FileNotFoundError:
        print("Plik nie został znaleziony.")
        return None
    except Exception as e:
        print("Wystąpił błąd podczas wczytywania tablicy numpy:", e)
        return None

def resize_flow_method(flow, new_size, step_size, method):
    assert flow is not None, "Flow data is None. Check if flow file was correctly loaded."
    
    h, w = new_size
    resized_flow = np.zeros((h, w, 2))
    
    H, W, _ = flow.shape
    for i in range(h):
        for j in range(w):
            x_start, y_start = i * step_size, j * step_size
            x_end, y_end = x_start + step_size, y_start + step_size
            block = flow[x_start:x_end, y_start:y_end]
            lengths = np.linalg.norm(block, axis=2)
            if method == 'max':
                index = np.unravel_index(np.argmax(lengths, axis=None), lengths.shape)
            elif method == 'median':
                sorted_indices = np.argsort(lengths, axis=None)
                median_index = sorted_indices[len(sorted_indices) // 2]
                index = np.unravel_index(median_index, lengths.shape)
            elif method == 'min':
                index = np.unravel_index(np.argmin(lengths, axis=None), lengths.shape)
            elif method == 'mean':
                mean_index = np.mean(block, axis=(0, 1))
                index = (block.shape[0] // 2, block.shape[1] // 2)  # Ustaw środek bloku jako indeks
            
            resized_flow[i, j, :] = block[index[0], index[1], :]

    return resized_flow

flow = read_flow_file('/mnt/d/datasety/autoflowaug/9sample_176/forward.npy')

# Przeprowadzenie procesu zmniejszania rozdzielczości
resized_max = resize_flow_method(flow, (18,32), 10, 'max')
resized_median = resize_flow_method(flow, (18,32), 10, 'median')
resized_min = resize_flow_method(flow, (18,32), 10, 'min')
resized_mean = resize_flow_method(flow, (18,32), 10, 'mean')

# Zapisanie wyników
#np.save('resized_max.npy', resized_max)
#np.save('resized_median.npy', resized_median)
#np.save('resized_min.npy', resized_min)
#np.save('resized_mean.npy', resized_mean)

# Wizualizacja pliku wejściowego
plt.imshow(flow_to_color(flow))
plt.title('Input Flow')
plt.axis('off')
plt.show()

# Wizualizacja przetworzonych plików
plt.imshow(flow_to_color(resized_max))
plt.title('Resized Max Flow')
plt.axis('off')
plt.show()

plt.imshow(flow_to_color(resized_median))
plt.title('Resized Median Flow')
plt.axis('off')
plt.show()

plt.imshow(flow_to_color(resized_min))
plt.title('Resized Min Flow')
plt.axis('off')
plt.show()

plt.imshow(flow_to_color(resized_mean))
plt.title('Resized Mean Flow')
plt.axis('off')
plt.show()



import os

def process_dataset(dataset_dir):
    total_files = 0
    processed_files = 0

    # Liczenie wszystkich plików do przetworzenia
    for subdir in os.listdir(dataset_dir):
        sub_dir_path = os.path.join(dataset_dir, subdir)
        if os.path.isdir(sub_dir_path):
            forward_path = os.path.join(sub_dir_path, 'forward.npy')
            if os.path.exists(forward_path):
                total_files += 1

    # Przetwarzanie wszystkich katalogów w dataset_dir
    for subdir in os.listdir(dataset_dir):
        sub_dir_path = os.path.join(dataset_dir, subdir)
        if os.path.isdir(sub_dir_path):
            forward_path = os.path.join(sub_dir_path, 'forward.npy')
            if os.path.exists(forward_path):
                flow = read_flow_file(forward_path)
                if flow is not None:
                    resized_max = np.transpose(resize_flow_method(flow, (18,32), 10, 'max'), (2, 0, 1))
                    resized_median = np.transpose(resize_flow_method(flow, (18,32), 10, 'median'), (2, 0, 1))
                    resized_min = np.transpose(resize_flow_method(flow, (18,32), 10, 'min'), (2, 0, 1))
                    resized_mean = np.transpose(resize_flow_method(flow, (18,32), 10, 'mean'), (2, 0, 1))

                    processed_flow = np.concatenate([resized_max, resized_median, resized_min, resized_mean], axis=0)

                    output_path = os.path.join(sub_dir_path, 'max_median_min_mean.npy')
                    np.save(output_path, processed_flow.astype(np.float16))

                    processed_files += 1
                    if processed_files % 2000 == 0:  # Wyświetlanie postępu co 1000 plików
                        print(f"Przetworzono {processed_files}/{total_files} plików.")

    print("Przetwarzanie zakończone.")

# Użycie funkcji do przetworzenia całego datasetu
dataset_dir = '/mnt/d/datasety/autoflowaug'
process_dataset(dataset_dir)
