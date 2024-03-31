# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        # img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L')            
        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img), np.array(mask, dtype=np.uint8)

class Compose2(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img1, img2):
        # img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L')
        assert img1.size == img2.size
        for a in self.augmentations:
            img1, img2 = a(img1, img2)
        return np.array(img1), np.array(img2)

class Compose_Pseudo(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img1, img2, mask_pseudo1, mask_pseudo2):
        # img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L')
        assert img1.size == img2.size == mask_pseudo1.size == mask_pseudo2.size
        for a in self.augmentations:
            img1, img2, mask_pseudo1, mask_pseudo2 = a(img1, img2, mask_pseudo1, mask_pseudo2)
        return np.array(img1), np.array(img2), np.array(mask_pseudo1, dtype=np.uint8), np.array(mask_pseudo2, dtype=np.uint8)


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

class RandomCrop2(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img1, img2):
        if self.padding > 0:
            img1 = ImageOps.expand(img1, border=self.padding, fill=0)
            img2 = ImageOps.expand(img2, border=self.padding, fill=0)

        assert img1.size == img2.size
        w, h = img1.size
        th, tw = self.size
        if w == tw and h == th:
            return img1, img2
        if w < tw or h < th:
            return img1.resize((tw, th), Image.BILINEAR), img2.resize((tw, th), Image.BILINEAR)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))

class RandomCrop_Pseudo(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img1, img2, mask_pseudo1, mask_pseudo2):
        if self.padding > 0:
            img1 = ImageOps.expand(img1, border=self.padding, fill=0)
            img2 = ImageOps.expand(img2, border=self.padding, fill=0)
            mask_pseudo1 = ImageOps.expand(mask_pseudo1, border=self.padding, fill=0)
            mask_pseudo2 = ImageOps.expand(mask_pseudo2, border=self.padding, fill=0)

        assert img1.size == img2.size == mask_pseudo1.size == mask_pseudo2.size
        w, h = img1.size
        th, tw = self.size
        if w == tw and h == th:
            return img1, img2, mask_pseudo1, mask_pseudo2
        if w < tw or h < th:
            return img1.resize((tw, th), Image.BILINEAR), img2.resize((tw, th), Image.BILINEAR), mask_pseudo1.resize((tw, th), Image.NEAREST), mask_pseudo2.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th)), mask_pseudo1.crop((x1, y1, x1 + tw, y1 + th)), mask_pseudo2.crop((x1, y1, x1 + tw, y1 + th))



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
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class RandomHorizontallyFlip2(object):
    def __call__(self, img1, img2):
        if random.random() < 0.5:
            return img1.transpose(Image.FLIP_LEFT_RIGHT), img2.transpose(Image.FLIP_LEFT_RIGHT)
        return img1, img2


class RandomHorizontallyFlip_Pseudo(object):
    def __call__(self, img1, img2, mask_pseudo1, mask_pseudo2):
        if random.random() < 0.5:
            return img1.transpose(Image.FLIP_LEFT_RIGHT), img2.transpose(Image.FLIP_LEFT_RIGHT), mask_pseudo1.transpose(Image.FLIP_LEFT_RIGHT), mask_pseudo2.transpose(Image.FLIP_LEFT_RIGHT)
        return img1, img2, mask_pseudo1, mask_pseudo2


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


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


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


class RandomSized_and_Crop(object):
    def __init__(self, size):
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.7, 1) * img.size[0])
        h = int(random.uniform(0.7, 1) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(img, mask)

class RandomSized_and_Crop2(object):
    def __init__(self, size):
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = RandomCrop2(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.7, 1) * img.size[0])
        h = int(random.uniform(0.7, 1) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.BILINEAR)

        return self.crop(img, mask)

class RandomSized_and_Crop_Pseudo(object):
    def __init__(self, size):
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = RandomCrop_Pseudo(self.size)

    def __call__(self, img1, img2, mask_pseudo1, mask_pseudo2):
        assert img1.size == img2.size == mask_pseudo1.size == mask_pseudo2.size

        w = int(random.uniform(0.7, 1) * img1.size[0])
        h = int(random.uniform(0.7, 1) * img1.size[1])

        img1, img2, mask_pseudo1, mask_pseudo2 = img1.resize((w, h),  Image.BILINEAR), img2.resize((w, h),  Image.BILINEAR), mask_pseudo1.resize((w, h), Image.NEAREST), mask_pseudo2.resize((w, h), Image.NEAREST)

        return self.crop(img1, img2, mask_pseudo1, mask_pseudo2)
