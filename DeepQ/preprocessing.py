from skimage import data, color, io
import numpy as np

def downsample(img_array):
    return img_array[::4, ::4]

def to_greyscale(img_array):
    return np.mean(img_array, axis=2).astype(np.uint8)

def preprocess(img_path):
    img_array = io.imread(img_path)
    return to_greyscale(downsample(img_array))