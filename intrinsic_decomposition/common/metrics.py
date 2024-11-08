import numpy as np
from skimage.segmentation import slic, mark_boundaries
import torch

EPSILON = 1e-6

def weighted_human_disagreement_rate(pred, gt, mask, total_points=10000, indices=None):
    # https://github.com/aim-uofa/AdelaiDepth/issues/26#issuecomment-1047483512
    if indices is None:
        p12_index = select_index(gt, mask=mask, select_size=total_points)
    else:
        p12_index = indices

    gt_reshape = np.reshape(gt, gt.size)
    pred_reshape = np.reshape(pred, pred.size)
    mask = np.reshape(mask, mask.size)
    gt_p1 = gt_reshape[mask][p12_index['p1']]
    gt_p2 = gt_reshape[mask][p12_index['p2']]
    pred_p1 = pred_reshape[mask][p12_index['p1']]
    pred_p2 = pred_reshape[mask][p12_index['p2']]

    p12_rank_gt = np.zeros_like(gt_p1)
    p12_rank_gt[gt_p1 > gt_p2] = 1
    p12_rank_gt[gt_p1 < gt_p2] = -1

    p12_rank_pred = np.zeros_like(gt_p1)
    p12_rank_pred[pred_p1 > pred_p2] = 1
    p12_rank_pred[pred_p1 < pred_p2] = -1

    err = np.sum(p12_rank_gt != p12_rank_pred)
    valid_pixels = gt_p1.size
    err = err / valid_pixels
    return err

def select_index(gt_depth, mask, select_size=10000):
    valid_size = mask.sum()
    try:
        p = np.random.choice(valid_size, select_size*2, replace=False)
    except:
        p = np.random.choice(valid_size, select_size*2*2, replace=True)
    np.random.shuffle(p)
    p1 = p[0:select_size*2:2]
    p2 = p[1:select_size*2:2]

    p12_index = {'p1': p1, 'p2': p2}
    return p12_index

def rmse_error(pred: np.ndarray, target: np.ndarray, mask:np.ndarray=None) -> float:
    """Root Mean Squared Error
    Args: 
        pred (np.ndarray): predicted values with dimension (H, W)
        target (np.ndarray): target values with dimension (H, W)
        mask (np.ndarray, optional): mask for the values with dimension (H, W)
    Returns:
        float: RMSE
    """
    
    mask = mask if mask is not None else np.ones_like(pred)

    pred = pred[mask]
    target = target[mask]

    diff = (pred - target) ** 2
    valid_pixels = np.sum(mask)
    error = np.sqrt(np.sum(diff) / (valid_pixels + EPSILON))

    return error

def absolute_relative_error(pred: np.ndarray, target: np.ndarray, mask:np.ndarray=None) -> float:
    """Absolute Relative Error
    Args: 
        pred (np.ndarray): predicted values with dimension (H, W)
        target (np.ndarray): target values with dimension (H, W)
        mask (np.ndarray, optional): mask for the values with dimension (H, W)
    Returns:
        float: ARE
    """
    
    mask = mask if mask is not None else np.ones_like(pred)

    pred = pred[mask]
    target = target[mask]

    diff = np.abs(pred - target) / (np.abs(target) + EPSILON)
    valid_pixels = np.sum(mask)
    error = np.sum(diff) / (valid_pixels + EPSILON)

    return error

def delta_error(pred: np.ndarray, target: np.ndarray, mask:np.ndarray=None):
    """Delta Error: https://github.com/YvanYin/DiverseDepth/blob/master/Train/lib/utils/evaluate_depth_error.py#L132
    Args: 
        pred (np.ndarray): predicted values with dimension (H, W)
        target (np.ndarray): target values with dimension (H, W)
        mask (np.ndarray, optional): mask for the values with dimension (H, W)
    """
    pred = pred[mask]
    target = target[mask]

    pred = pred.reshape(1, -1)
    target = target.reshape(1, -1)

    ratio = np.concatenate((pred/target, target/pred), axis=0)
    max_ratio = np.amax(ratio, axis=0)

    perc = (np.sum((max_ratio < 1.25).astype(np.float32)) / np.sum(mask)) * 100
    return perc


def ssq_error(correct, estimate, mask):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate**2 * mask) > 1e-5:
        alpha = np.sum(correct * estimate * mask) / np.sum(estimate**2 * mask)
    else:
        alpha = 0.
    return np.sum(mask * (correct - alpha*estimate) ** 2)

def lmse(correct, estimate, mask, window_size, window_shift):
    if len(correct.shape) == 2 or correct.shape[-1] == 1:
        return lmse_gray(correct, estimate, mask, window_size, window_shift)
    else:
        return lmse_rgb(correct, estimate, mask, window_size, window_shift)

def lmse_rgb(correct, estimate, mask, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N = correct.shape[:2]
    ssq = total = 0.

    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):

            correct_curr = correct[i:i+window_size, j:j+window_size, :]
            estimate_curr = estimate[i:i+window_size, j:j+window_size, :]
            mask_curr = mask[i:i+window_size, j:j+window_size]

            rep_mask = np.concatenate([mask_curr] * 3, 0)
            rep_cor = np.concatenate([correct_curr[:, :, 0], correct_curr[:, :, 1], correct_curr[:, :, 2]], 0)
            rep_est = np.concatenate([estimate_curr[:, :, 0], estimate_curr[:, :, 1], estimate_curr[:, :, 2]], 0)

            ssq += ssq_error(rep_cor, rep_est, rep_mask)
            # FIX: in the original codebase, this was outdented, which allows
            # for scores greater than 1 (which should not be possible).  On the
            # MIT dataset images, this makes a negligible difference, but on
            # larger images, this can have a significant effect.
            total += np.sum(rep_mask * rep_cor**2)

    assert ~np.isnan(ssq/total)

    return ssq / total

def lmse_gray(correct, estimate, mask, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N = correct.shape[:2]
    ssq = total = 0.

    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):

            correct_curr = correct[i:i+window_size, j:j+window_size]
            estimate_curr = estimate[i:i+window_size, j:j+window_size]
            mask_curr = mask[i:i+window_size, j:j+window_size]
            ssq += ssq_error(correct_curr, estimate_curr, mask_curr)
            # FIX: in the original codebase, this was outdented, which allows
            # for scores greater than 1 (which should not be possible).  On the
            # MIT dataset images, this makes a negligible difference, but on
            # larger images, this can have a significant effect.
            total += np.sum(mask_curr * correct_curr**2)

    assert ~np.isnan(ssq/total)

    return ssq / total

def lmse_downscale(correct, estimate, mask, window_size, window_shift, downscale):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N = correct.shape[:2]
    ssq = total = 0.

    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):
            
            correct_curr = correct[i:i+window_size, j:j+window_size]
            estimate_curr = estimate[i:i+window_size, j:j+window_size]
            mask_curr = mask[i:i+window_size, j:j+window_size]
            
            res_cor_curr = cv2.resize(correct_curr, (downscale, downscale))
            res_est_curr = cv2.resize(estimate_curr, (downscale, downscale))
            
            res_msk_curr = cv2.resize(mask_curr.astype(np.float32), (downscale, downscale), cv2.INTER_NEAREST)
            res_msk_curr = res_msk_curr.astype(bool)
            
            ssq += ssq_error(res_cor_curr, res_est_curr, res_msk_curr)
            # FIX: in the original codebase, this was outdented, which allows
            # for scores greater than 1 (which should not be possible).  On the
            # MIT dataset images, this makes a negligible difference, but on
            # larger images, this can have a significant effect.
            total += np.sum(res_msk_curr * res_cor_curr**2)
            
    assert ~np.isnan(ssq/total)

    return ssq / total


def compute_grad(img):
    # img = gaussian_filter(img, 0.5)
    # show(img)
    dy = sobel(img, axis=0)
    dx = sobel(img, axis=1)
    return np.stack([dx, dy], axis=-1)

def ssq_grad_error(correct, estimate, mask):

    assert correct.ndim == 2
    
    # the mask is (h, w, 2) to compare gradients, 
    # but sometimes we need the (h, w) version..
    single_mask = mask[:, :, 0]
    
    if np.sum(estimate**2 * single_mask) > 1e-5:
        alpha = np.sum(correct * estimate * single_mask) / np.sum(estimate**2 * single_mask)
    else:
        alpha = 0.
        
    scaled_est = alpha * estimate
    est_grad_mag = compute_grad(scaled_est)
    cor_grad_mag = compute_grad(correct)
    
    return np.sum(mask * (cor_grad_mag - est_grad_mag) ** 2),  cor_grad_mag

def grad_lmse(correct, estimate, mask, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N = correct.shape[:2]
    ssq = total = 0.

    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):
            
            correct_curr = correct[i:i+window_size, j:j+window_size]
            estimate_curr = estimate[i:i+window_size, j:j+window_size]
            mask_curr = mask[i:i+window_size, j:j+window_size]
            
            # repeat mask to create a two channel image
            mask_curr = np.stack([mask_curr] * 2, -1)

            error, corr_grad = ssq_grad_error(correct_curr, estimate_curr, mask_curr)
            ssq += error

            # FIX: in the original codebase, this was outdented, which allows
            # for scores greater than 1 (which should not be possible).  On the
            # MIT dataset images, this makes a negligible difference, but on
            # larger images, this can have a significant effect.
            total += np.sum(mask_curr * corr_grad**2)
            
    assert ~np.isnan(ssq/total)

    return ssq / total


EPSILON = 1e-6

def run_slic(gtdisp, nsamples, compactness=1):
    segments = slic(gtdisp, n_segments=nsamples, compactness=compactness, start_label=0, slic_zero=True)

    segments_ids = np.unique(segments)

    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    seg_img = mark_boundaries(gtdisp, segments, color=(0.0, 0.1, 0.9))

    point_pairs = []
    for i in range(bneighbors.shape[1]):
        if bneighbors[0][i] != bneighbors[1][i]:
            point_pairs.append((bneighbors[0][i], bneighbors[1][i]))

    return centers, point_pairs, seg_img

def fast_ordering_ratio(base, point_loc1, point_loc2, threshold):

    ratio = base[point_loc1[:, 0], point_loc1[:, 1]] / base[point_loc2[:, 0], point_loc2[:, 1]]

    out = np.zeros_like(ratio)

    out[ratio > (1.0 + threshold)] = 1
    out[ratio < (1.0 - threshold)] = -1
    return out

def fast_ordering_diff(base, point_loc1, point_loc2, threshold):
    diff = base[point_loc1[:, 0], point_loc1[:, 1]] - base[point_loc2[:, 0], point_loc2[:, 1]]

    out = np.zeros_like(diff)

    out[diff > threshold] = 1
    out[diff < -threshold] = -1
    return out

def fast_ordering(base, point_loc1, point_loc2, threshold, mode):
    if mode == 'ratio':
        return fast_ordering_ratio(base, point_loc1, point_loc2, threshold)
    elif mode == 'diff':
        return fast_ordering_diff(base, point_loc1, point_loc2, threshold)

def fast_d3r(pred, target, freq_threshold, threshold, nsamples, mode='diff', debug=False, mask=None, compactness=1.0, slic=None):
    """ Computing D3R metric using diff instead of ratio to compute the ordinal relations
    Args:
        gtdisp ([torch array]): Ground truth disparity map between 0,1 not containing Nan values.
        preddisp ([torch array]): Prediction disparity map between 0,1 not containing Nan values.
        freq_threshold ([float]): A threshold to define high frequecy changes
        threshold ([float]): Threshold to define ordinal relations based on diff
        nsamples ([int]): Number of superpixels created by SLIC alghorithm
    Returns:
        [float, list of tuples, list of tuples, 2darray]: computed error value, list of selected point pairs,
         list of point pairs that had mismatching orders, position of centers of superpixels
    """
    EPSILON = 1e-6

    gtdisp = target
    preddisp = pred
    mask = mask if mask is not None else np.ones_like(preddisp)

    if slic is None:
        centers, point_pairs, seg_img = run_slic(
            gtdisp,
            nsamples,
            compactness=compactness
        )

    else:
        centers = slic['centers']
        point_pairs = slic['point_pairs']

    not_matching_pairs = []
    selected_pairs = []

    np_pt_pairs = np.array(point_pairs)
    center1_locs = np.floor(centers[np_pt_pairs[:, 0]]).astype(np.int32)
    center2_locs = np.floor(centers[np_pt_pairs[:, 1]]).astype(np.int32)

    # mask out pairs where either/both are outside mask
    mask_loc1 = mask[center1_locs[:, 0], center1_locs[:, 1]].astype(bool)
    mask_loc2 = mask[center2_locs[:, 0], center2_locs[:, 1]].astype(bool)
    valid = mask_loc1 & mask_loc2

    # mask out pairs where there isn't a sufficient edge
    gt_freq_ord = fast_ordering(gtdisp, center1_locs, center2_locs, freq_threshold, mode)
    valid &= (gt_freq_ord != 0)

    grnd_ordering = fast_ordering(gtdisp, center1_locs, center2_locs, threshold, mode)
    pred_ordering = fast_ordering(preddisp, center1_locs, center2_locs, threshold, mode)

    # not_matching = (grnd_ordering != pred_ordering)
    valid = valid.astype(np.uint8)

    # d3r_error = abs(grnd_ordering - pred_ordering) * valid
    d3r_error = (grnd_ordering != pred_ordering).astype(np.uint8) * valid
    d3r_error = d3r_error.sum() / (valid.sum() + EPSILON)

    if debug:
        correct = (grnd_ordering == pred_ordering)
        return {
            # 'seg_img' : seg_img,
            'valid_mask' : valid,
            'centers_a' : center1_locs,
            'centers_b' : center2_locs,
            'correct' : correct,
            'error' : d3r_error
        }

    return d3r_error

def compute_whdr(reflectance, judgements, delta=0.10):
    """
    NOTE (chris): I stripped this directly from the file that is provided
    as part of the IIW dataset
    Return the WHDR score for a reflectance image, evaluated against human
    judgements.  The return value is in the range 0.0 to 1.0, or None if there
    are no judgements for the image.  See section 3.5 of our paper for more
    details.
    :param reflectance: a numpy array containing the linear RGB
    reflectance image.
    :param judgements: a JSON object loaded from the Intrinsic Images in
    the Wild dataset.
    :param delta: the threshold where humans switch from saying "about the
    same" to "one point is darker."
    """

    points = judgements['intrinsic_points']
    comparisons = judgements['intrinsic_comparisons']
    id_to_points = {p['id']: p for p in points}
    rows, cols = reflectance.shape[0:2]

    error_sum = 0.0
    weight_sum = 0.0

    for c in comparisons:
        # "darker" is "J_i" in our paper
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        # "darker_score" is "w_i" in our paper
        weight = c['darker_score']
        if weight <= 0 or weight is None:
            continue

        point1 = id_to_points[c['point1']]
        point2 = id_to_points[c['point2']]
        if not point1['opaque'] or not point2['opaque']:
            continue

        # convert to grayscale and threshold
        l1 = max(1e-10, np.mean(reflectance[
            int(point1['y'] * rows), int(point1['x'] * cols), ...]))
        l2 = max(1e-10, np.mean(reflectance[
            int(point2['y'] * rows), int(point2['x'] * cols), ...]))

        # convert algorithm value to the same units as human judgements
        if l2 / l1 > 1.0 + delta:
            alg_darker = '1'
        elif l1 / l2 > 1.0 + delta:
            alg_darker = '2'
        else:
            alg_darker = 'E'

        if darker != alg_darker:
            error_sum += weight
        weight_sum += weight

    if weight_sum:
        return error_sum / weight_sum
    else:
        return None
