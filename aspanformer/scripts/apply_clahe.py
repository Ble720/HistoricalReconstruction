import cv2
import os
import numpy as np
import argparse

def mean(img):
    return round(np.mean(img), 3)

def std(img):
    return round(np.std(img), 3)

def applyclahe(img, name):
    if mean(img) < 75:
        clache = cv2.createCLAHE(clipLimit=16.0)
        img = clache.apply(img)
        print(f'{name} clip 16')
    elif std(img) < 25:
        clache = cv2.createCLAHE(clipLimit=4.0)
        img = clache.apply(img)
        print(f'{name} clip 4')
    elif mean(img) < 150 and std(img) < 50:
        clache = cv2.createCLAHE(clipLimit=1.5)
        img = clache.apply(img)
        print(f'{name} clip 1.5')
    return img

def main(path, save):
    for p in os.listdir(path):
        img = cv2.imread(f'{path}/{p}', cv2.IMREAD_GRAYSCALE)
        img = applyclahe(img, p)
        cv2.imwrite(f'{save}/{p}', img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to source images')
    parser.add_argument('-s', '--save', help='save path')
    args = parser.parse_args()

    if not os.path.isdir(args.save):
        os.makedirs(args.save, exist_ok=True)
    main(args.path, args.save)