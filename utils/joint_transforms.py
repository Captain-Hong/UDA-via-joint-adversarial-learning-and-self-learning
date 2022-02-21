# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:22:14 2020

@author: 11627
"""
# joint_transforms
import cv2
import math
import sys
import numbers
import random
from PIL import Image, ImageOps
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import helpers


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(math.ceil((w - tw) / 2.))
        y1 = int(math.ceil((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class SingleCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        x1 = int(math.ceil((w - tw) / 2.))
        y1 = int(math.ceil((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop_npy(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.shape == mask.shape
        if (self.size <= img.shape[1]) and (self.size <= img.shape[0]):
            x = math.ceil((img.shape[1] - self.size) / 2.)
            y = math.ceil((img.shape[0] - self.size) / 2.)

            if len(mask.shape) == 3:
                return img[y:y + self.size, x:x + self.size, :], mask[y:y + self.size, x:x + self.size, :]
            else:
                return img[y:y + self.size, x:x + self.size, :], mask[y:y + self.size, x:x + self.size]
        else:
            raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
                self.size, self.size, img.shape[0], img.shape[1]))

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size=0, scale_rate=0.95, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_rate = scale_rate
        self.fill = fill

    def __call__(self, im, gt):
        img = im.copy()
        mask = gt.copy()
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * self.scale_rate), int(self.base_size * self.scale_rate))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

