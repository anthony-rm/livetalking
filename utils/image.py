
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# def read_imgs(img_list):
#     frames = []
#     logger.info('reading images...')
#     for img_path in tqdm(img_list):
#         frame = cv2.imread(img_path)
#         frames.append(frame)
#     return frames

def read_imgs(img_list):
    def load_image(index, img_path):
        return index, cv2.imread(img_path)

    frames = [None] * len(img_list)  # Initialize a list with the same length as img_list
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_image, idx, img_path): idx for idx, img_path in enumerate(img_list)}
        for future in tqdm(as_completed(futures), total=len(img_list)):
            idx, img = future.result()
            frames[idx] = img
    return frames

def mirror_index(size, index):
    if size <= 0:
        # An empty frame list (e.g. a MuseTalk avatar driven through the
        # wav2lip path, which has no face_imgs/) makes inference_batch crash
        # with ZeroDivisionError. The inference thread dies silently, the
        # pipeline stalls, and the whole server hangs. Treat it as a single
        # frame so callers get a clear failure instead of a silent hang.
        return 0
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1 