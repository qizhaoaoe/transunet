from torch.utils.data import Dataset
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
import glob


def z_score_norm(img):
    u = np.mean(img)
    s = np.std(img)
    img -= u
    if s == 0:
        return img
    return img/s


def min_max_norm(img, epsilon=1e-5):
    minv = np.min(img)
    maxv = np.max(img)
    return (img - minv + epsilon) / (maxv - minv + epsilon)


def get_img_label_paths(data_file):
    img_label_plist = []
    with open(data_file, 'r') as f:
        for l in f:
            img_label_plist.append(l.strip().split(','))
    return img_label_plist


def resize_3d(img, resize_shape, order=0):
    zoom0 = resize_shape[0] // img.shape[0]
    zoom1 = resize_shape[1] // img.shape[1]
    zoom2 = resize_shape[2] // img.shape[2]
    img = zoom(img, (zoom0, zoom1, zoom2), order=order)
    return img


def get_img(img_path):
    img_itk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_itk)
    return img


def save_img(img, save_path):
    img_itk = sitk.GetImageFromArray(img)
    sitk.WriteImage(img_itk, save_path)


class MyDataset(Dataset):
    def __init__(self, data_dir, data_file, input_shape, transforms=None, target_transforms=None):
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.img_label_plist = get_img_label_paths(data_file)
        self.input_shape = input_shape
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        x_path, y_path = self.img_label_plist[index]
        img_x = get_img(os.path.join(self.data_dir, x_path)).astype(np.float32)
        img_y = get_img(os.path.join(self.data_dir, y_path)).astype(np.float32)
        img_x = z_score_norm(img_x)
        img_x = min_max_norm(img_x)
        img_x = resize_3d(img_x, self.input_shape, 1)
        img_y = resize_3d(img_y, self.input_shape)
        img_x = np.expand_dims(img_x, 0)
        img_y[img_y > 1] = 0
        if self.transforms is not None:
            img_x = self.transforms(img_x)
        if self.target_transforms is not None:
            img_y = self.target_transforms(img_y)
        return img_x, img_y

