import PIL
from PIL import Image, ImageOps, ImageEnhance
import random
import matplotlib.pyplot as plt


def shear_x(image, value):  # [-0.3: 0.3]
    assert -0.3 <= value <= 0.3
    if random.random() > 0.5:
        value = -value

    return image.transform(image.size, PIL.Image.AFFINE, (1, value, 0, 0, 1, 0))


def shear_y(image, value):  # [-0.3: 0.3]
    assert -0.3 <= value <= 0.3
    if random.random() > 0.5:
        value = -value

    return image.transform(image.size, PIL.Image.AFFINE, (1, 0, 0, value, 1, 0))


def translate_x(image, value):  # [-0.45, 0.45]
    assert -0.45 <= value <= 0.45
    if random.random() > 0.5:
        value = -value

    value *= image.size[0]

    return image.transform(image.size, PIL.Image.AFFINE, (1, 0, value, 0, 1, 0))


def translate_y(image, value):  # [-0.45, 0.45]
    assert -0.45 <= value <= 0.45
    if random.random() > 0.5:
        value = -value

    value *= image.size[1]

    return image.transform(image.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, value))


def rotate(image, value):  # [-30, 30]
    assert -30 <= value <= 30
    if random.random() > 0.5:
        value = -value

    return image.rotate(value)


def auto_contrast(image, _):

    return ImageOps.autocontrast(image)


def invert(image, _):

    return ImageOps.invert(image)


def equalize(image, _):

    return ImageOps.equalize(image)


def flip(image, _):

    return ImageOps.mirror(image)


def solarize(image, _):

    return ImageOps.solarize(image)


def posterize(image, value):  # [1, 8]
    assert 1 <= value <= 8
    value = int(value)
    value = max(1, value)

    return ImageOps.posterize(image, value)


def contrast(image, value):  # [0.1: 1.9]
    assert 0.1 <= value <= 1.9

    return ImageEnhance.Contrast(image).enhance(value)


def color(image, value):  # [0.1: 1.9]
    assert 0.1 <= value <= 1.9

    return ImageEnhance.Color(image).enhance(value)


def brightness(image, value):  # [0.1: 1.9]
    assert 0.1 <= value <= 1.9

    return ImageEnhance.Brightness(image).enhance(value)


def sharpness(image, value):  # [0.1: 1.9]
    assert 0.1 <= value <= 1.9

    return ImageEnhance.Sharpness(image).enhance(value)


def identity(image, _):

    return image


def augment_list():
    aug_list = [
        (auto_contrast, 0, 1),
        # (equalize, 0, 1),
        # (invert, 0, 1),
        (rotate, -30, 30),
        # (posterize, 1, 8),
        # (solarize, 0, 256),
        (color, 0.1, 1.9),
        # (contrast, 0.1, 1.9),
        # (brightness, 0.1, 1.9),
        (sharpness, 0.1, 1.9),
        (shear_x, -0.3, 0.3),
        (shear_y, -0.3, 0.3),
        (translate_x, -0.45, 0.45),
        (translate_y, -0.45, 0.45)
    ]

    return aug_list


class RandAugment(object):
    def __init__(self, n, m):
        self.num_select = n
        self.magnitude = m
        self.augment_list = augment_list()

    def __call__(self, image):
        ops = random.choices(self.augment_list, k=self.num_select)
        for op, min_value, max_value in ops:
            val = (float(self.magnitude) / 30) * (max_value - min_value) + min_value
            image = op(image, val)
        return image
    

if __name__ == "__main__":
    path = 'data/raw/public_test/000.jpg'
    image = Image.open(path)
    # pass image to RandAugment
    rand = RandAugment(3,5)
    image = rand(image)
    plt.imshow(image)
    plt.show()