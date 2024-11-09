import numpy as np
from PIL import Image
import random
import yaml
import datetime

import math
import torch
import torchvision.transforms.functional as TF



def tonemap(img):
    return img/(img+1.0)


def current_timestamp():
    return datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')

def load_config(path) -> dict:
    with open(path, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)
    return cfg

def read_image(path):
    return np.array(Image.open(path)).astype(np.float32) / 255.

def read_ldr_image(path):
    return np.array(Image.open(path))

# these are the sRGB <-> linear functions from CGIntrinsics and Luo
def rgb_to_srgb(rgb):
    ret = np.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = np.power(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret

def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret

# from Bjoern Ottoson @ https://bottosson.github.io/posts/oklab/
class Lab:
    def __init__(self, L, a, b):
        self.L = L
        self.a = a
        self.b = b

class RGB:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

def linear_srgb_to_oklab(c):
    l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b
    m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b
    s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b

    l_ = math.pow(l, 1/3)
    m_ = math.pow(m, 1/3)
    s_ = math.pow(s, 1/3)

    return Lab(
        0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_,
        1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_,
        0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_
    )

def oklab_to_linear_srgb(c):
    l_ = c.L + 0.3963377774 * c.a + 0.2158037573 * c.b
    m_ = c.L - 0.1055613458 * c.a - 0.0638541728 * c.b
    s_ = c.L - 0.0894841775 * c.a - 1.2914855480 * c.b

    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    return RGB(
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    )

def LAB_to_numpy(lab):
    return np.array([lab.L, lab.a, lab.b])

def RGB_to_numpy(rgb):
    return np.array([rgb.r, rgb.g, rgb.b])

def LAB_to_pytorch(lab):
    return torch.tensor([lab.L, lab.a, lab.b])

def RGB_to_pytorch(rgb):
    return torch.tensor([rgb.r, rgb.g, rgb.b])

# invert and revert shading
def real_to_inv(shading):
    inv_shading = 1/(shading + 1.0)
    return inv_shading

def inv_to_real(inv_shading):
    shading = (1.0 / inv_shading) - 1.0
    return shading



def sample_1d(img, y_idx):
    b, h, c = img.size()  # [b, h, c]
    b, n = y_idx.size()    # [b, n]
    
    b_idx = torch.arange(b, dtype=torch.int64, device=img.device).unsqueeze(-1)  # [b, 1]
    b_idx = b_idx.expand(b, n)  # [b, n]
    
    y_idx = torch.clamp(y_idx, 0, h - 1)  # [b, n]
    a_idx = torch.stack([b_idx, y_idx], dim=-1)  # [b, n, 2]
    
    return torch.gather(img, 1, a_idx[..., 1].unsqueeze(-1).expand(b, n, c))


def interp_1d(img, y):
    b, h, c = img.size()  # [b, h, c]
    b, n = y.size()        # [b, n]
    
    y_0 = torch.floor(y)  # [b, n]
    y_1 = y_0 + 1    
    
    _sample_func = lambda y_x: sample_1d(img, y_x.int())
    y_0_val = _sample_func(y_0)  # [b, n, c]
    y_1_val = _sample_func(y_1)
    
    w_0 = y_1 - y  # [b, n]
    w_1 = y - y_0
    
    w_0 = w_0.unsqueeze(-1)  # [b, n, 1]
    w_1 = w_1.unsqueeze(-1)
    
    return w_0 * y_0_val + w_1 * y_1_val


def apply_rf(x, rf):
    b, *s = x.size()  # [b, s...]
    b, k = rf.size()   # [b, k]
    x = interp_1d(
        rf.unsqueeze(-1),                                   # [b, k, 1] 
        (k - 1) * x.view(b, -1).to(torch.float32),  # [b, ?] 
    )  # [b, ?, 1]
    return x.view(b, *s)



def random_color_jitter(img):

    hue_shft = (random.randint(0, 50) / 50.) - 0.5
    hue_img = TF.adjust_hue(img, hue_shft)
    
    sat_shft = (random.randint(0, 50) / 50.) + 0.5
    sat_img = TF.adjust_saturation(hue_img, sat_shft)

    r_mul = 1.0 + (random.randint(0, 100) / 250) - 0.2
    b_mul = 1.0 + (random.randint(0, 100) / 250) - 0.2
    sat_img[0, :, :] *= r_mul
    sat_img[2, :, :] *= b_mul

    return sat_img

def random_crop_and_resize(images, output_size=384,  min_crop=128):
    _, h, w = images.shape
    
    max_crop = min(h, w)
    
    rand_crop = random.randint(min_crop, max_crop)
    
    rand_top = random.randint(0, h - rand_crop)
    rand_left = random.randint(0, w - rand_crop)
    
    images = TF.crop(images, rand_top, rand_left, rand_crop, rand_crop)
    images = TF.resize(images, (output_size, output_size), antialias=True)

    return images

def random_flip(images, p=0.5):
    if random.random() > p:
        return TF.hflip(images)
    else:
        return images