import os
from idlelib.pyparse import trans

import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def resize_train(image, size):
    resize = transforms.Resize(size)
    return resize(image)


def resize_label(image, size):
    resize = transforms.Resize(size, interpolation=InterpolationMode.NEAREST)
    return resize(image)

def histogram_equal(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    l, a, b = cv2.split(img)

    lab_clahe = cv2.merge((cv2.equalizeHist(l), a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_YCrCb2RGB)

    return result

def clahe(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    l, a, b = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_YCrCb2RGB)

    return result

def retinex_MSRCR(img, sigma_list=[5, 15, 30], gain=1.0, offset=0):
    # Dataset : sigma_list = [5, 15, 30], gain=1.0, offset=0
    # Dataset3 : sigma_list = [15, 80, 250], gain=1.0, offset=0
    img = img.astype(np.float32) + 1.0
    log_R = np.zeros_like(img)

    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        log_R += np.log(img) - np.log(blur + 1.0)

    log_R /= len(sigma_list)

    sum_channels = np.sum(img, axis=2, keepdims=True)
    crf = np.log(img / (sum_channels + 1e-6) + 1.0)


    msrcr = gain * log_R * crf + offset

    msrcr = np.clip(msrcr, 0, None)
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX)
    msrcr = np.clip(msrcr, 0, 255).astype(np.uint8)

    return msrcr

def retinex_MSR(img, sigma_list= [5, 15, 30]):
    # Dataset : sigma_list = [5, 15, 30]
    # Dataset3 : sigma_list = [15, 80, 250]
    img = img.astype(np.float32) + 1.0
    result = np.zeros_like(img)
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        result += np.log(img) - np.log(blur + 1)
    result = result / len(sigma_list)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(result)


def gammacorrection(img, gamma=0.5):
    # if Night Dataset used set gamma=2.0
    # if weekly Dataset used set gamma=0.5
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)


class vehicledata():
    CLASSES = (
        'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
        'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
        'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
        'rubberCone', 'trafficSign', 'warningTriangle', 'fence'
    )

    def __init__(self, image_path, annotation_path, n_class, size, transform=None):
        self.image_path = image_path
        self.train_dir = sorted(os.listdir(self.image_path))
        #
        self.annotation_path = annotation_path
        self.ann_file = sorted(os.listdir(self.annotation_path))
        #
        self.size = size
        self.n_class = n_class

    def __len__(self):
        return len(self.train_dir)

    def __getitem__(self, index):
        #
        assert self.train_dir[index].split('.')[0] == self.ann_file[index].split('.')[0], f'file names are different...'

        # Training_image
        img = os.path.join(self.image_path, self.train_dir[index])
        img = Image.open(img)

        # Label
        label = os.path.join(self.annotation_path + self.ann_file[index])
        label = Image.open(label)

        img = resize_train(img, self.size)
        label = resize_label(label, self.size)

        img = np.array(img, dtype=np.uint8)
        label = np.array(label, dtype=np.uint8)

        img, label = self.transform(img, label)

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1


        return img, target, label, index

    def transform(self, img, label):
        img = img[:, :, ::-1]  # RGB -> BGR

        # img = histogram_equal(img)

        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float() / 255.0    # when image is into network -> image pixel value is between 0 and 1
        label = torch.from_numpy(label).float()
        # normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # img = normalize(img)

        return img, label

    def untransform(self, img, label):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img = img.astype(np.uint8)
        img = img[:, :, ::-1] * 255.0
        label = label.numpy()

        return img, label


if __name__ == "__main__":
    image_path = "/storage/sjpark/vehicle_data/Dataset3/train_image/"
    annotation_path = "/storage/sjpark/vehicle_data/Dataset3/ann_train/"
    dataset_object = vehicledata(image_path, annotation_path, 21, (256, 256))

    img, target, label,  index = dataset_object.__getitem__(20)
