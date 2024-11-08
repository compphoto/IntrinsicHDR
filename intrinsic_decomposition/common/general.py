import os
import sys
import numpy as np
import math
import skimage
import cv2
import scipy as sp
import scipy.ndimage

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from .data_util import np_to_pil

def pad_bb(bb, amount=7):
    return (
        bb[0] - 7,
        bb[1] - 7,
        bb[2] + 7,
        bb[3] + 7
    )

def add_chan(img):
    return np.stack([img] * 3, -1)

def _tile_row(images, text, border, font, font_pos, font_color, text_box):

    assert(len(images) == len(text))

    border_pix = np.ones((images[0].shape[0], border, 3))
    elements = [border_pix]

    for im, txt in zip(images, text):
        if len(im.shape) == 2:
            im = np.stack([im] * 3, -1)

        if im.shape[-1] == 1:
            im = np.concatenate([im] * 3, -1)
       
        if txt !=  "" and txt is not None:
            pil_im = np_to_pil(im)
            draw = ImageDraw.Draw(pil_im, "RGBA")

            txtbb = draw.textbbox(font_pos, txt, font=font)
        
            if text_box is not None:
                draw.rectangle(pad_bb(txtbb), fill=(0, 0, 0, text_box))

            if font_color is None:
                imgbb = im[txtbb[1]:txtbb[3], txtbb[0]:txtbb[2], :]
                brightness = imgbb.mean(-1).mean()
                font_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)

            draw.text(font_pos, txt, font_color, font=font)
            im = np.array(pil_im) / 255.

        elements.append(im)
        elements.append(border_pix)

    concat = np.concatenate(elements, axis=1)
    return concat

def tile_imgs(images, text=None, font_size=16, font_file="/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              font_color=None, font_pos=(0, 0), text_box=None, display=False, save=None, border=20, cmap='viridis', quality=75):
    
    if not isinstance(images, list):
        print("expected list of images")
        return
    
    # make 1d array in 2d to keep logic simple
    if not isinstance(images[0], list):
        images = [images]
        
    # if text is none make a 2d array like images to make things
    # easier, otherwise text should already be shaped like images
    if text is None:
        text = []
        for row in images:
            text.append([None] * len(row))
    else:
        if not isinstance(text[0], list):
            text = [text]
        
    font = ImageFont.truetype(font_file, font_size)

    width = sum([border + x.shape[1] for x in images[0]]) + border
    border_pix = np.ones((border, width, 3))
    rows = [border_pix]
    
    for img_row, txt_row in zip(images, text):
        tiled_row = _tile_row(img_row, txt_row, border, font, font_pos, font_color, text_box)
        
        rows.append(tiled_row)
        rows.append(border_pix)

    tiled = np.concatenate(rows, axis=0)

    if display:
        plt.figure(figsize=(len(images[0]), len(images)))
        plt.imshow(tiled, cmap=cmap)
        plt.axis('off')

    if save:
        # TODO: save tiled image
        byte_img = (tiled * 255.).astype(np.uint8)
        Image.fromarray(byte_img).save(save, quality=quality)
        # pass

    return tiled

def show(img, size=(16, 9), save=None):
    if isinstance(img, list):
        img = tile_imgs(img, save=save)

    plt.figure(figsize=size)
    plt.imshow(img)
    plt.axis('off')

def match_scale(pred, grnd, mask=None, skip_close=False, threshold=0.001):
    if mask is None:
        mask = np.ones(pred.shape[:2]).astype(bool)

    flat_pred = pred[mask].reshape(-1)
    flat_grnd = grnd[mask].reshape(-1)

    scale, _, _, _ = np.linalg.lstsq(flat_pred.reshape(-1, 1), flat_grnd, rcond=None)
    
    if skip_close and abs(1.0 - scale) < threshold:
        return scale

    return scale * pred

def get_brightness(rgb, mode='numpy'):

    # "CCIR601 YIQ" method for computing brightness
    if mode == 'numpy':
        brightness = (0.3 * rgb[:,:,0]) + (0.59 * rgb[:,:,1]) + (0.11 * rgb[:,:,2])
        return brightness[:, :, np.newaxis]
    if mode == 'torch':
        brightness = (0.3 * rgb[0,:,:]) + (0.59 * rgb[1,:,:]) + (0.11 * rgb[2, :,:])
        return brightness.unsqueeze(0)

def minmax(img):
    return (img - img.min()) / img.max()

def inv_2_real(inv_shd):
    # first normalize the network inverse shading to be [0,1]
    norm_inv_shd = minmax(inv_shd)

    # convert to regular shading, clip very small values for division
    shd = (1.0 / norm_inv_shd.clip(1e-5)) - 1.0

    return shd.clip(1e-5)

def round_32(x):
    return 32 * math.ceil(x / 32)

def to2np(img):
    return img.detach().cpu().permute(1, 2, 0).numpy()

def view_scale(img, p=100):
    return (img / np.percentile(img, p)).clip(0, 1)

def view(img, perc=100):
    return view_scale(img ** (1/2.2), p=perc)

def to2np(img):
    return img.detach().cpu().permute(1, 2, 0).numpy()

# GUIDED FILTER
def box(img, r):
    """ O(1) box filter
        img - >= 2d image
        r   - radius of box filter
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)


    tile = [1] * img.ndim
    tile[0] = r
    imCum = np.cumsum(img, 0)
    imDst[0:r+1, :, ...] = imCum[r:2*r+1, :, ...]
    imDst[r+1:rows-r, :, ...] = imCum[2*r+1:rows, :, ...] - imCum[0:rows-2*r-1, :, ...]
    imDst[rows-r:rows, :, ...] = np.tile(imCum[rows-1:rows, :, ...], tile) - imCum[rows-2*r-1:rows-r-1, :, ...]

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1, ...] = imCum[:, r:2*r+1, ...]
    imDst[:, r+1:cols-r, ...] = imCum[:, 2*r+1 : cols, ...] - imCum[:, 0 : cols-2*r-1, ...]
    imDst[:, cols-r: cols, ...] = np.tile(imCum[:, cols-1:cols, ...], tile) - imCum[:, cols-2*r-1 : cols-r-1, ...]

    return imDst

def _gf_color(I, p, r, eps, s=None):
    """ Color guided filter
    I - guide image (rgb)
    p - filtering input (single channel)
    r - window radius
    eps - regularization (roughly, variance of non-edge noise)
    s - subsampling factor for fast guided filter
    """
    fullI = I
    fullP = p
    if s is not None:
        I = sp.ndimage.zoom(fullI, [1/s, 1/s, 1], order=1)
        p = sp.ndimage.zoom(fullP, [1/s, 1/s], order=1)
        r = round(r / s)

    h, w = p.shape[:2]
    N = box(np.ones((h, w)), r)

    mI_r = box(I[:,:,0], r) / N
    mI_g = box(I[:,:,1], r) / N
    mI_b = box(I[:,:,2], r) / N

    mP = box(p, r) / N

    # mean of I * p
    mIp_r = box(I[:,:,0]*p, r) / N
    mIp_g = box(I[:,:,1]*p, r) / N
    mIp_b = box(I[:,:,2]*p, r) / N

    # per-patch covariance of (I, p)
    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    # symmetric covariance matrix of I in each patch:
    #       rr rg rb
    #       rg gg gb
    #       rb gb bb
    var_I_rr = box(I[:,:,0] * I[:,:,0], r) / N - mI_r * mI_r;
    var_I_rg = box(I[:,:,0] * I[:,:,1], r) / N - mI_r * mI_g;
    var_I_rb = box(I[:,:,0] * I[:,:,2], r) / N - mI_r * mI_b;

    var_I_gg = box(I[:,:,1] * I[:,:,1], r) / N - mI_g * mI_g;
    var_I_gb = box(I[:,:,1] * I[:,:,2], r) / N - mI_g * mI_b;

    var_I_bb = box(I[:,:,2] * I[:,:,2], r) / N - mI_b * mI_b;

    a = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            sig = np.array([
                [var_I_rr[i,j], var_I_rg[i,j], var_I_rb[i,j]],
                [var_I_rg[i,j], var_I_gg[i,j], var_I_gb[i,j]],
                [var_I_rb[i,j], var_I_gb[i,j], var_I_bb[i,j]]
            ])
            covIp = np.array([covIp_r[i,j], covIp_g[i,j], covIp_b[i,j]])
            a[i,j,:] = np.linalg.solve(sig + eps * np.eye(3), covIp)

    b = mP - a[:,:,0] * mI_r - a[:,:,1] * mI_g - a[:,:,2] * mI_b

    meanA = box(a, r) / N[...,np.newaxis]
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, [s, s, 1], order=1)
        meanB = sp.ndimage.zoom(meanB, [s, s], order=1)

    q = np.sum(meanA * fullI, axis=2) + meanB

    return q


def _gf_gray(I, p, r, eps, s=None):
    """ grayscale (fast) guided filter
        I - guide image (1 channel)
        p - filter input (1 channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    if s is not None:
        Isub = sp.ndimage.zoom(I, 1/s, order=1)
        Psub = sp.ndimage.zoom(p, 1/s, order=1)
        r = round(r / s)
    else:
        Isub = I
        Psub = p


    (rows, cols) = Isub.shape

    N = box(np.ones([rows, cols]), r)

    meanI = box(Isub, r) / N
    meanP = box(Psub, r) / N
    corrI = box(Isub * Isub, r) / N
    corrIp = box(Isub * Psub, r) / N
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP


    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = box(a, r) / N
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, s, order=1)
        meanB = sp.ndimage.zoom(meanB, s, order=1)

    q = meanA * I + meanB
    return q


def _gf_colorgray(I, p, r, eps, s=None):
    """ automatically choose color or gray guided filter based on I's shape """
    if I.ndim == 2 or I.shape[2] == 1:
        return _gf_gray(I, p, r, eps, s)
    elif I.ndim == 3 and I.shape[2] == 3:
        return _gf_color(I, p, r, eps, s)
    else:
        print("Invalid guide dimensions:", I.shape)


def guided_filter(I, p, r, eps, s=None):
    """ run a guided filter per-channel on filtering input p
        I - guide image (1 or 3 channel)
        p - filter input (n channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    if p.ndim == 2:
        p3 = p[:,:,np.newaxis]
    else:
        p3 = p

    out = np.zeros_like(p3)
    for ch in range(p3.shape[2]):
        out[:,:,ch] = _gf_colorgray(I, p3[:,:,ch], r, eps, s)

    return np.squeeze(out) if p.ndim == 2 else out

