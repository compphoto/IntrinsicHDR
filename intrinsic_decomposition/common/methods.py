import os
import sys
import math
from skimage.transform import resize

import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from .general import round_32, get_brightness, to2np, view_scale
from .data_util import load_image, np_to_pil

COMPARISONS_DIR = ''

# -------------------------------------------------------------------
def round_128(x):
    return 128 * math.ceil(x / 128)

def luo_resize(srgb_img): 
 
    ratio = float(srgb_img.shape[0])/float(srgb_img.shape[1]) 
 
    if ratio > 1.73: 
        h, w = 320, 160 #512, 256 
    elif ratio < 1.0/1.73: 
        h, w = 160, 320 #256, 512 
    elif ratio > 1.41: 
        h, w = 384, 256 
    elif ratio < 1./1.41: 
        h, w = 256, 384 
    elif ratio > 1.15: 
        h, w = 320, 240 #512, 384 
    elif ratio < 1./1.15: 
        h, w = 240, 320 #384, 512 
    else: 
        h, w = 320, 320 #384, 384 
 
    srgb_img = resize(srgb_img, (h, w), order=1, preserve_range=True) 
 
    return srgb_img 

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

# -------------------------------------------------------------------
def run_li_2018_eccv(img, orig_resize=False, linear_input=False, linear_output=False, verbose=False):
    root_dir = f'{COMPARISONS_DIR}/li_2018_eccv/'

    h, w, _ = img.shape
    
    # if we have a linear input make it sRGB otherwise leave it
    if linear_input:
        img = rgb_to_srgb(img)
    
    # if we aren't using the original resizing code, round
    # to a multiple of 128 to make sure there are not size issues
    if not orig_resize:
        new_h, new_w = round_128(h), round_128(w)
        img = resize(img, (new_h, new_w))

    np_to_pil(img).save(f'{root_dir}/input/img.png')

    cmd = f'cd {root_dir} && python decompose.py'
    
    if orig_resize:
        cmd += ' --resize'
    
    if not verbose:
        cmd += ' > /dev/null 2>&1'

    os.system(cmd)

    alb = load_image(f'{root_dir}/output/ref.png')
    shd = load_image(f'{root_dir}/output/shd.png')
    
    rec = alb * shd[:, :, None]
    
    # if the output should be linear, leave it, otherwise make it sRGB
    if not linear_output:
        alb = rgb_to_srgb(alb)
        shd = rgb_to_srgb(shd)
        rec = rgb_to_srgb(rec)
    
    alb = resize(alb, (h, w), anti_aliasing=True)
    shd = resize(shd, (h, w), anti_aliasing=True)
    rec = resize(rec, (h, w), anti_aliasing=True)

    return alb, shd, rec

def run_li_2018_cvpr(img, orig_resize=False, linear_input=False, linear_output=False, verbose=False):
    root_dir = f'{COMPARISONS_DIR}/li_2018_cvpr/'

    h, w, _ = img.shape
    
    # if we have a linear input make it sRGB otherwise leave it
    if linear_input:
        img = rgb_to_srgb(img)
    
    # if we aren't using the original resizing code, round
    # to a multiple of 128 to make sure there are not size issues
    if not orig_resize:
        new_h, new_w = round_128(h), round_128(w)
        img = resize(img, (new_h, new_w))

    np_to_pil(img).save(f'{root_dir}/input/img.png')

    cmd = f'cd {root_dir} && python decompose.py'
    
    if orig_resize:
        cmd += ' --resize'
    
    if not verbose:
        cmd += ' > /dev/null 2>&1'

    os.system(cmd)

    alb = load_image(f'{root_dir}/output/ref.png')
    shd = load_image(f'{root_dir}/output/shd.png')
    
    rec = alb * shd
    
    # if the output should be linear, leave it, otherwise make it sRGB
    if not linear_output:
        alb = rgb_to_srgb(alb)
        shd = rgb_to_srgb(shd)
        rec = rgb_to_srgb(rec)
    
    alb = resize(alb, (h, w), anti_aliasing=True)
    shd = resize(shd, (h, w), anti_aliasing=True)
    rec = resize(rec, (h, w), anti_aliasing=True)

    return alb, shd, rec

def run_luo_2020(img, orig_resize=False, linear_input=False, linear_output=False, verbose=False):
    sys.path.append(f'{COMPARISONS_DIR}/')

    import os
    from os import listdir

    import torch
    from PIL import Image
    from torchvision.transforms import functional as TF

    from luo_2020.config import TestOptions
    from luo_2020.model.manager import create_model
    from luo_2020.utils import image_util
    from luo_2020.utils import pytorch_settings

    opt = TestOptions()
    opt.parse({
        'pretrained_file' : f'{COMPARISONS_DIR}/luo_2020/pretrained_model/final.pth.tar'
    })
    # print(kwargs)

    # torch setting
    # pytorch_settings.set_(with_random=False, determine=True)

    # visualize
    # V.create_a_visualizer(opt)

    # NIID-Net Manager
    model = create_model(opt)
    model.switch_to_eval()

    h, w, _ = img.shape
    
    # if we have a linear input make it sRGB otherwise leave it
    if linear_input:
        img = rgb_to_srgb(img)
    
    # if specified use the resizing function from the luo code-base
    # otherwise round to a multiple of 32 to avoid size issues 
    if orig_resize:
        img = luo_resize(img)
    else:
        img = resize(img, (round_32(h), round_32(w)), anti_aliasing=True)

    input_img = TF.to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        pred_N, pred_R, pred_L, pred_S, rendered_img = model.predict({'input_srgb': input_img}, normal=True, IID=True)

    alb = to2np(pred_R[0])
    shd = to2np(pred_S[0])
    rec = to2np(rendered_img[0])
    
    # if the output should be linear, leave it, otherwise make it sRGB
    # view_scale only scales between 0-1 using max value
    # view first calls view_scale and then gamma corrects using 2.2
    alb = view_scale(alb)
    shd = view_scale(shd[:, :, 0])
    rec = view_scale(rec)
    
    if not linear_output:
        alb = rgb_to_srgb(alb)
        shd = rgb_to_srgb(shd)
        rec = rgb_to_srgb(rec)
    
    alb = resize(alb, (h, w), anti_aliasing=True)
    shd = resize(shd, (h, w), anti_aliasing=True)
    rec = resize(rec, (h, w), anti_aliasing=True)

    return alb, shd, rec

def run_liu_2020(img, orig_resize=False, linear_input=False, linear_output=False, verbose=False):
    root_dir = f'{COMPARISONS_DIR}/liu_2020/'

    h, w, _ = img.shape
    
    # if we have a linear input make it sRGB otherwise leave it
    if linear_input:
        img = rgb_to_srgb(img)

    np_to_pil(img).save(f'{root_dir}/input/img.png')

    cmd = f'cd {root_dir} && python3 test.py -c configs/intrinsic_MIX_IIW.yaml -i input -o output -p pretrained_model/gen-MIX.pt'
        
    if orig_resize:
        cmd += ' --resize'
    
    if not verbose:
        cmd += ' > /dev/null 2>&1'

    os.system(cmd)

    alb = load_image(f'{root_dir}/output/img/output_r.jpg')
    shd = load_image(f'{root_dir}/output/img/output_s.jpg')
    shd = shd[:, :, 0]

    rec = alb * shd[:, :, None]
    
    # if the output should be linear, leave it, otherwise make it sRGB
    if not linear_output:
        alb = rgb_to_srgb(alb)
        shd = rgb_to_srgb(shd)
        rec = rgb_to_srgb(rec)
    
    alb = resize(alb, (h, w), anti_aliasing=True)
    shd = resize(shd, (h, w), anti_aliasing=True)
    rec = resize(rec, (h, w), anti_aliasing=True)

    return alb, shd, rec

def run_vanhoey_2018(img, orig_resize=False, linear_input=False, linear_output=False, verbose=False):
    root_dir = f'{COMPARISONS_DIR}/vanhoey_2018/'

    h, w, _ = img.shape
    
    # if we have a linear input make it sRGB otherwise leave it
    if linear_input:
        img = rgb_to_srgb(img)

    np_to_pil(img).save(f'{root_dir}/input/img.png')

    cmd = f'cd {root_dir} && bash decompose.sh'

    if not verbose:
        cmd += ' > /dev/null 2>&1'

    os.system(cmd)

    alb = load_image(f'{root_dir}/output/img_A.png')
    shd = load_image(f'{root_dir}/output/img_S.png')
    
    rec = alb * shd
    
    if linear_output:
        alb = srgb_to_rgb(alb)
        shd = srgb_to_rgb(shd)
        rec = srgb_to_rgb(rec)
        
    alb = resize(alb, (h, w), anti_aliasing=True)
    shd = resize(shd, (h, w), anti_aliasing=True)
    rec = resize(rec, (h, w), anti_aliasing=True)

    return alb, shd, rec

def run_baslamisli_2018_eccv(img, orig_resize=False, linear_input=False, linear_output=False, verbose=False):
    root_dir = f'{COMPARISONS_DIR}/baslamisli_2018_eccv/'

    h, w, _ = img.shape
    
    # if we have a linear input make it sRGB otherwise leave it
    if linear_input:
        img = rgb_to_srgb(img)

    # if we aren't using the original resizing code, round
    # to a multiple of 32 to make sure there are not size issues
    if not orig_resize:
        new_h, new_w = round_32(h), round_32(w)
        img = resize(img, (new_h, new_w))

    np_to_pil(img).save(f'{root_dir}/input/img.png')

    cmd = f'cd {root_dir} && python infer.py --name synthetic_trained --file input/img.png --gpu 0 --model ./'
    
    if orig_resize:
        cmd += ' --resize'
    
    if not verbose:
        cmd += ' > /dev/null 2>&1'

    os.system(cmd)

    alb = load_image(f'{root_dir}/results/disn/synthetic_trained/images/albedo/input/img.png')
    shd = load_image(f'{root_dir}/results/disn/synthetic_trained/images/shading/input/img.png')[:, :, 0]
    
    rec = alb * shd[:, :, None]

    if linear_output:
        alb = srgb_to_rgb(alb)
        shd = srgb_to_rgb(shd)
        rec = srgb_to_rgb(rec)
        
    alb = resize(alb, (h, w), anti_aliasing=True)
    shd = resize(shd, (h, w), anti_aliasing=True)
    rec = resize(rec, (h, w), anti_aliasing=True)

    return alb, shd, rec

def run_das_2022(img, orig_resize=False, linear_input=False, linear_output=False, verbose=False, gpu=True):
    root_dir = f'{COMPARISONS_DIR}/das_2022/'

    h, w, _ = img.shape
    
    # if we have a linear input make it sRGB otherwise leave it
    if linear_input:
        img = rgb_to_srgb(img)
    
    if max(h, w) > 1000 and not orig_resize:
        gpu = False

    if not orig_resize:
        new_h, new_w = round_32(h), round_32(w)
        img = resize(img, (new_h, new_w))

    np_to_pil(img).save(f'{root_dir}/input/img.png')

    cmd = f'cd {root_dir} && python decompose.py'
        
    if orig_resize:
        cmd += ' --resize'
    if gpu:
        cmd += ' --gpu'

    if not verbose: 
        cmd += ' > /dev/null 2>&1'

    os.system(cmd)

    alb = load_image(f'{root_dir}/output/out_pred_alb.png')
    shd = load_image(f'{root_dir}/output/out_pred_shd.png')
    
    rec = alb * shd[:, :, None]

    if linear_output:
        alb = srgb_to_rgb(alb)
        shd = srgb_to_rgb(shd)
        rec = srgb_to_rgb(rec)
    
    alb = resize(alb, (h, w), anti_aliasing=True)
    shd = resize(shd, (h, w), anti_aliasing=True)
    rec = resize(rec, (h, w), anti_aliasing=True)

    return alb, shd, rec

def run_baslamisli_2018_cvpr(img, orig_resize=False, linear_input=False, linear_output=False, verbose=False):
    # NOTE (chris): for this method we always use the orig_resize code because
    # the model can't accept any size input only the specific original size

    root_dir = f'{COMPARISONS_DIR}/baslamisli_2018_cvpr/intrinsicNet'

    h, w, _ = img.shape

    # if we have a linear input make it sRGB otherwise leave it
    if linear_input:
        img = rgb_to_srgb(img)

    np_to_pil(img).save(f'{root_dir}/input/img.png')

    cmd = f'cd {root_dir} && bash decompose.sh'

    if not verbose:
        cmd += ' > /dev/null 2>&1'

    os.system(cmd)

    alb = load_image(f'{root_dir}/output/alb.png')
    shd = load_image(f'{root_dir}/output/shd.png')
    
    rec = alb * shd[:, :, None]

    if linear_output:
        alb = srgb_to_rgb(alb)
        shd = srgb_to_rgb(shd)
        rec = srgb_to_rgb(rec)
        
    alb = resize(alb, (h, w), anti_aliasing=True)
    shd = resize(shd, (h, w), anti_aliasing=True)
    rec = resize(rec, (h, w), anti_aliasing=True)

    return alb, shd, rec


def run_shen_2011(img, orig_resize=False, linear_input=False, linear_output=False, verbose=False):
    root_dir = f'{COMPARISONS_DIR}/shen_2011/'

    np_to_pil(img).save(f'{root_dir}/input/img.png')

    if linear_input:
        lin = 1
    else:
        lin = 0

    cmd = f"matlab -nodisplay -nosplash -nodesktop -r \"addpath('{root_dir}'); decompose('{root_dir}/input/img.png', '{root_dir}/output/', {lin});exit;\""

    if not verbose:
        cmd += '> /dev/null 2>&1'

    os.system(cmd)

    alb = load_image(f'{root_dir}/output/reflectance.png')
    shd = load_image(f'{root_dir}/output/shading.png')
    rec = alb * shd[:, :, None]

    if not linear_output:
        alb = rgb_to_srgb(alb)
        shd = rgb_to_srgb(shd)
        rec = rgb_to_srgb(rec)

    return alb, shd, rec

def run_bell_2014(img, orig_resize=False, linear_input=False, linear_output=False, verbose=False):
    root_dir = f'{COMPARISONS_DIR}/bell_2014/'
    
    np_to_pil(img).save(f'{root_dir}/input/img.png')

    cmd = f'cd {root_dir} && bash decompose.sh'
    
    if linear_input:
        cmd += ' --linear'

    if not verbose:
        cmd += '> /dev/null 2>&1'

    os.system(cmd)

    alb = load_image(f'{root_dir}/output/ref.png')
    shd = load_image(f'{root_dir}/output/shd.png')
    rec = alb * shd[:, :, None]
        
    if linear_input and not linear_output:
        alb = rgb_to_srgb(alb)
        shd = rgb_to_srgb(shd)
        rec = rgb_to_srgb(rec)
    
    if linear_output and not linear_input:
        alb = srgb_to_rgb(alb)
        shd = srgb_to_rgb(shd)
        rec = srgb_to_rgb(rec)


    return alb, shd, rec
