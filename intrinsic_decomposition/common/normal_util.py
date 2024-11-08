import torch
import numpy as np
import cv2

def prediction_to_normal(pred):
    # x, y = pred[:, 0, :, :], pred[:, 1, :, :]
    # z = torch.sqrt((1.0 - (x**2 + y**2)).clip(1e-4))

    # return torch.stack((x, y, z), dim=1)
    x, y, z = pred[:, 0, :, :], pred[:, 1, :, :], pred[:, 2, :, :]

    x = torch.tanh(x)
    y = torch.tanh(y)
    z = torch.sigmoid(z)

    pred = torch.stack((x, y, z), dim=1)

    norm = torch.linalg.norm(pred, dim=1, keepdim=True).clip(1e-4)
    return pred / norm

def angular_error(gt, pred, mask):
    gt = gt.astype(np.float64) 
    pred = pred.astype(np.float64) 

    gt /= np.linalg.norm(gt, axis=-1, keepdims=True).clip(1e-4)
    pred /= np.linalg.norm(pred, axis=-1, keepdims=True).clip(1e-4)
    
    # compute the vector product between gt and prediction for each pixel
    dot_prod = (gt * pred).sum(axis=-1)

    # compute the angle from vector product
    ang_err = np.arccos(np.clip(dot_prod, -1.0, 1.0))

    # convert radius to degrees
    ang_err *= (180 / np.pi)

    return ang_err[mask]

def compute_metrics(ang_err):
    # six surface normal metrics following: https://web.eecs.umich.edu/~fouhey/2016/evalSN/evalSN.html

    mean_err = np.mean(ang_err)
    med_err = np.median(ang_err)
    rmse = np.sqrt(np.mean(np.power(ang_err, 2.0)))

    t1 = np.mean(ang_err < 11.25)
    t2 = np.mean(ang_err < 22.5)
    t3 = np.mean(ang_err < 30.0)

    return mean_err, med_err, rmse, t1, t2, t3
   
def depth_to_normals(depth, k=7, perc=90):
    # depth: input depth as np.array
    # k: sobel kernel size for depth gradient computation
    # perc: percentile used to clip outliers in gradient magnitude

    h, w = depth.shape
    
    # compute x and y gradients with sobel
    x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=k)     
    y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=k)

    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    
    # get the magnitude of all the gradients
    grad_mag = (flat_x ** 2) + (flat_y ** 2)

    # get the x-th percentile value (the rest are outliers)
    max_val = np.percentile(grad_mag, perc, interpolation='nearest')
    max_pos = np.where(grad_mag == max_val)[0][0]

    # get the x and y values of the point with max gradient
    max_x, max_y = flat_x[max_pos], flat_y[max_pos]
    
    # compute the scalar c, such that the max gradient position has a z-value of zero
    c_2 = 1.0 / ((max_x ** 2) + (max_y ** 2))
    c = np.sqrt(c_2)
    
    # get the value of the magnitudes after scaling by c
    scaled_mags = (((flat_x*c) ** 2) + ((flat_y*c) ** 2))

    # get the magnitude of the scaled gradients
    # at the max_pos it should be very close to 1
    scaled_max = scaled_mags[max_pos]
    assert(np.isclose(scaled_max, 1.0))

    # any magnitudes that were huge (past 90th percentile)
    # can be clipped to the same value as max_pos (1)
    scaled_mags = scaled_mags.clip(0, 1)

    # now we can compute the z value as the value that makes
    # each pixel have a magnitude of 1
    z = 1.0 - scaled_mags
    z = z.reshape(h, w)
    
    # stack the components and then normalize, this will 
    # scale down the outliers whose magnitudes we clipped
    normal = np.stack((-x, y, z), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    return normal 
