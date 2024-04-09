import numpy as np
import cv2
import matplotlib.pyplot as plt

def flow_to_color(flow):
    """
    Konwertuje wektor przepływu optycznego do kolorowego obrazu.
    :param flow: Wektor przepływu optycznego o wymiarach (H, W, 2).
    :return: Kolorowy obraz przepływu optycznego o wymiarach (H, W, 3).
    """
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
            
            resized_flow[i, j, :] = block[index[0], index[1], :]

    return resized_flow

flow = read_flow_file('/mnt/d/datasety/autoflowaug/9sample_376/forward.npy')

# Przeprowadzenie procesu zmniejszania rozdzielczości
resized_max = resize_flow_method(flow, (18,32), 10, 'max')
resized_median = resize_flow_method(flow, (18,32), 10, 'median')
resized_min = resize_flow_method(flow, (18,32), 10, 'min')

# Zapisanie wyników
#np.save('resized_max.npy', resized_max)
#np.save('resized_median.npy', resized_median)
#np.save('resized_min.npy', resized_min)

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
