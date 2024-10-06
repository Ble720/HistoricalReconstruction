import os
import cv2
import numpy as np
import torch

def get_paths(dir):
    all_path = []
    for fRoot, fDirs, fFiles in os.walk(dir):
        for ffile in fFiles:
            if ffile.endswith('.jpg') or ffile.endswith('.jpeg'):
                full_path = os.path.join(fRoot, ffile).replace('/', os.sep)
                all_path.append(full_path)
    return all_path

def read_image_gray(path, resize):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    scale = resize / max(h,w)
    hnew, wnew = int(h*scale), int(w*scale)
    image = cv2.resize(image, (hnew, wnew))
    padded_image = torch.zeros((resize, resize))
    padded_image[:hnew, :wnew] = image
    return padded_image, [hnew, wnew]

def check_matches(mkpts0, mkpts1, mask0, mask1, b_ids, batch_size):
    keep = torch.le(mkpts0, mask0)
    keep = keep[:, 0] * keep[:, 1]
    new_b_ids = b_ids[keep]
    keep = torch.le(mkpts1[keep, :], mask1[new_b_ids, :])
    keep = keep[:, 0] * keep[:, 1]
    new_b_ids = new_b_ids[keep]
    return torch.bincount(new_b_ids, minlength=batch_size)