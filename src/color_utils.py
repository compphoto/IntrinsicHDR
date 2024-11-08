import torch
import numpy as np
import kornia as kn



def rgb_to_lab(rgb, normalize=True,mode='torch'):

    if mode=='numpy':
        rgb = torch.tensor(rgb).permute(2,0,1).unsqueeze(0)

    # adapted from kornia without linearization
    xyz_im =  kn.color.rgb_to_xyz(rgb)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x = xyz_int[..., 0, :, :]
    y = xyz_int[..., 1, :, :]
    z = xyz_int[..., 2, :, :]

    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    _b = 200.0 * (y - z)

    lab = torch.stack([L, a, _b], dim=-3)

    if normalize:
        # normalize
        lab[:,0,:,:] = lab[:,0,:,:]/100.0
        lab[:,1:,:,:] = (lab[:,1:,:,:]+128.0)/255.0

    if mode=='numpy':
        lab = lab.squeeze(0).permute(1,2,0).numpy()

    return lab