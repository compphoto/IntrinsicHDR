import sys
import skimage
import cv2
import numpy as np

sys.path.append('../')
from .general import round_32

def resizewithpool(img, size):
    i_size = img.shape[0]
    n = int(np.floor(i_size/size))

    out = skimage.measure.block_reduce(img, (n, n), np.max)
    return out

def rgb2gray(rgb):
    # Converts rgb to gray
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def calculateprocessingres(img, basesize, confidence=0.1, scale_threshold=3, whole_size_threshold=3000):
    # Returns the R_x resolution described in section 5 of the main paper.

    # Parameters:
    #    img :input rgb image
    #    basesize : size the dilation kernel which is equal to receptive field of the network.
    #    confidence: value of x in R_x; allowed percentage of pixels that are not getting any contextual cue.
    #    scale_threshold: maximum allowed upscaling on the input image ; it has been set to 3.
    #    whole_size_threshold: maximum allowed resolution. (R_max from section 6 of the main paper)

    # Returns:
    #    outputsize_scale*speed_scale :The computed R_x resolution
    #    patch_scale: K parameter from section 6 of the paper

    # speed scale parameter is to process every image in a smaller size to accelerate the R_x resolution search
    speed_scale = 32
    image_dim = int(min(img.shape[0:2]))

    gray = rgb2gray(img)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad = cv2.resize(grad, (image_dim, image_dim), cv2.INTER_AREA)

    # thresholding the gradient map to generate the edge-map as a proxy of the contextual cues
    m = grad.min()
    M = grad.max()
    middle = m + (0.4 * (M - m))
    grad[grad < middle] = 0
    grad[grad >= middle] = 1

    # dilation kernel with size of the receptive field
    kernel = np.ones((int(basesize/speed_scale), int(basesize/speed_scale)), np.float64)
    # dilation kernel with size of the a quarter of receptive field used to compute k
    # as described in section 6 of main paper
    kernel2 = np.ones((int(basesize / (4*speed_scale)), int(basesize / (4*speed_scale))), np.float64)

    # Output resolution limit set by the whole_size_threshold and scale_threshold.
    threshold = min(whole_size_threshold, scale_threshold * max(img.shape[:2]))
    if threshold < basesize:
        print("WARNING: common/resolution_util.py | the image size is too small - Using the receptive field size as the output resolution.")
        return basesize, 1

    outputsize_scale = basesize / speed_scale
    for p_size in range(int(basesize/speed_scale), int(threshold/speed_scale), int(basesize / (2*speed_scale))):
        grad_resized = resizewithpool(grad, p_size)
        grad_resized = cv2.resize(grad_resized, (p_size, p_size), cv2.INTER_NEAREST)
        grad_resized[grad_resized >= 0.5] = 1
        grad_resized[grad_resized < 0.5] = 0

        dilated = cv2.dilate(grad_resized, kernel, iterations=1)
        meanvalue = (1-dilated).mean()
        if meanvalue > confidence:
            break
        else:
            outputsize_scale = p_size

    grad_region = cv2.dilate(grad_resized, kernel2, iterations=1)
    patch_scale = grad_region.mean()

    return int(outputsize_scale*speed_scale), patch_scale

def get_optimal_dims(img, conf):
    h, w, _ = img.shape 

    # get the larger of the two dimensions and determine the scale 
    max_dim = max(h, w)
    
    if conf<0:
        # use the boosting depth code to get the optimal resolution for 0 confidence
        opt_size, _ = calculateprocessingres(img, 384, confidence=0)
        # substract conf from 100% to get new optimal size
        opt_size = opt_size*(1+conf)
    else:
        # use the boosting depth code to get the optimal resolution
        opt_size, _ = calculateprocessingres(img, 384, confidence=conf)
    
    # cap it between current img dim and maximum size
    opt_size = min(max(opt_size, max_dim), 6000)
   
    # resize by to make the longer side the optimal size
    scale = opt_size / max_dim
    
    # make sure the image is going to fit through the model correctly
    opt_h, opt_w = round_32(h * scale), round_32(w * scale)

    return opt_h, opt_w

def optimal_resize(img, conf=0.01):

    opt_h, opt_w = get_optimal_dims(img, conf) 
    print(f'Processing in {opt_w} by {opt_h}.')

    return skimage.transform.resize(img, (opt_h, opt_w), order=1, preserve_range=True)

