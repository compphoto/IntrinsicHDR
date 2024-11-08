import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import random
from PIL import Image
import numpy as np
import cv2
import skimage
import torchvision.transforms.functional as TF
import requests
import urllib
from io import BytesIO
from bs4 import BeautifulSoup

def load_from_url(url):
    response = requests.get(url)
    return load_image(BytesIO(response.content))

def load_image(path):
    return np.array(Image.open(path)).astype(np.float32) / 255.

def load_hdr(path):
    return (cv2.imread(path, cv2.IMREAD_ANYCOLOR)/(2**16)).astype(np.float32)

def load_depth(path):
    depth = Image.open(path)
    depth_arr = np.array(depth).astype(np.float32)
    depth_arr = depth_arr / (2**16)

    return depth_arr.astype(np.float32)

def get_tonemap_scale(rgb_color, p=90):
    gamma                             = 1.0 / 2.2   # standard gamma correction exponent
    inv_gamma                         = 1.0 / gamma
    # percentile                        = 90        # we want this percentile brightness value in the unmodified image...
    brightness_nth_percentile_desired = 0.8       # ...to be this bright after scaling

    brightness       = get_brightness(rgb_color)
    # brightness_valid = brightness[valid_mask]

    eps                               = 0.0001 # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
    brightness_nth_percentile_current = np.percentile(brightness, p)

    if brightness_nth_percentile_current < eps:
        scale = 0.0
    else:
        # Snavely uses the following expression in the code at https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
        # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma - np.log(brightness_nth_percentile_current))
        #
        # Our expression below is equivalent, but is more intuitive, because it follows more directly from the expression:
        # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired
        scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

    return scale

def get_brightness(rgb, mode='numpy'):

    # "CCIR601 YIQ" method for computing brightness
    if mode == 'numpy':
        brightness = (0.3 * rgb[:,:,0]) + (0.59 * rgb[:,:,1]) + (0.11 * rgb[:,:,2])
        return brightness[:, :, np.newaxis]
    if mode == 'torch':
        brightness = (0.3 * rgb[0,:,:]) + (0.59 * rgb[1,:,:]) + (0.11 * rgb[2, :,:])
        return brightness.unsqueeze(0)

def np_to_pil(img):
    """ converts a [0-1] numpy array into a PIL image """
    int_img = (img * 255).astype(np.uint8)
    return Image.fromarray(int_img)

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
    _, h, w = images[0].shape
    
    max_crop = min(h, w)
    
    rand_crop = random.randint(min_crop, max_crop)
    
    rand_top = random.randint(0, h - rand_crop)
    rand_left = random.randint(0, w - rand_crop)
    
    images = [TF.crop(x, rand_top, rand_left, rand_crop, rand_crop) for x in images]
    images = [TF.resize(x, (output_size, output_size)) for x in images]

    return images

def random_flip(images, p=0.5):
    if random.random() > p:
        return [TF.hflip(x) for x in images]
    else:
        return images

def get_main_img(soup):
    for entry in soup.findAll('a'):
        link = entry.get('href')
        if link is None: continue

        if 'http://labelmaterial.s3.amazonaws.com/photos/' in link:
            return link

def load_opensurfaces_image(id_num):
    
    IMAGE_PATH = '/home/chris/research/intrinsic/data/opensurfaces/images'

    url = f'http://opensurfaces.cs.cornell.edu/photos/{id_num}/'
    r = requests.get(url)
    content = r.content
    
    soup = BeautifulSoup(content)
    
    img_url = get_main_img(soup)
    
    img_base = img_url.split('/')[-1]
    img_path = os.path.join(IMAGE_PATH, img_base)

    # check if the image is already downloaded
    if not os.path.exists(img_path):
        print("downloading original image file")
        urllib.request.urlretrieve(img_url, img_path)

    img_arr = load_image(img_path)

    h, w, _ = img_arr.shape
    print(h, w)
    
    return img_arr, soup
