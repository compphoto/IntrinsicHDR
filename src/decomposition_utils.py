import numpy as np
import torch
from src.color_utils import rgb_to_lab
import torchvision.transforms.functional as TF
from intrinsic_decomposition.common.general import round_32, get_brightness


def get_quantile(img, thresh):

    if img.numel()>=16000000:
        qtl = torch.tensor(np.quantile(img.detach().cpu().numpy(), q=thresh)).to(img.device)
    else:
        qtl = torch.quantile(img,thresh)

    return qtl

def equalize_predictions(img, base, full, p=0.5):

    b,c,h, w = img.shape

    full_shd = (1. / full.clamp(1e-5)) - 1.
    base_shd = (1. / base.clamp(1e-5)) - 1.

    full_alb = get_brightness(img.squeeze(0),mode='torch') / full_shd.clamp(1e-5)
    base_alb = get_brightness(img.squeeze(0),mode='torch') / base_shd.clamp(1e-5)

    rand_msk = torch.randn(h,w) > p
    
    flat_full_alb = full_alb[:,:,(rand_msk == 1)]
    flat_base_alb = base_alb[:,:,(rand_msk == 1)]


    try:
        scale, _, _, _ = torch.linalg.lstsq(flat_full_alb.reshape(b,-1,1), flat_base_alb.reshape(b,-1,1), rcond=None)
        scale = scale.unsqueeze(3)
        success=True
    except Exception as e:
        print(e)
        success=False
        scale=1


    new_full_alb = scale * full_alb
    new_full_shd = get_brightness((img / new_full_alb.clamp(1e-5)).squeeze(0),mode='torch').unsqueeze(0)
    new_full = 1.0 / (1.0 + new_full_shd)

    return base, new_full,success


def decompose_torch(models, img_arr,resize_res=None,base_size=384,lstsq_p=0.0):
    _,_,orig_h, orig_w = img_arr.shape
    
    if resize_res == None:
        img_arr = TF.resize(img_arr, (round_32(orig_h), round_32(orig_w)), antialias=True)

    else:
        scale = resize_res / max(orig_h, orig_w)
        img_arr = TF.resize(img_arr, (round_32(orig_h * scale), round_32(orig_w * scale)), antialias=True)


    _,_,fh, fw = img_arr.shape
    
    lin_img = img_arr.to(models['ordinal_model'].device)
        
    with torch.no_grad():
        # ordinal shading estimation --------------------------
        max_dim = max(fh, fw)
        scale = base_size / max_dim
        new_h, new_w = scale * fh, scale * fw
        new_h, new_w = round_32(new_h), round_32(new_w)

        base_input = TF.resize(lin_img,(new_h,new_w),antialias=True)
        full_input = lin_img

        base_out = models['ordinal_model'](base_input.float())
        full_out = models['ordinal_model'](full_input.float())
        

        base_out = TF.resize(base_out, (fh, fw),antialias=True).unsqueeze(1)
        full_out = full_out.unsqueeze(1)
        
        ord_base, ord_full,success = equalize_predictions(lin_img, base_out, full_out, p=lstsq_p)
        if success==False:
            return None,None,success

        
        combined = torch.cat((lin_img, ord_base, ord_full),dim=1)
        inv_shd = models['real_model'](combined.float()).unsqueeze(1)
        
        shd = ((1.0 / inv_shd) - 1.0)
        alb = lin_img / shd
        # ------------------------------------------------------
                      
    inv_shd = TF.resize(inv_shd, (orig_h, orig_w), antialias=True)
    alb = TF.resize(alb, (orig_h, orig_w), antialias=True)
    
    return inv_shd,alb


def blend_albedo(albedo_low,albedo_high,overlap_mask,alpha):
    lab_alb_shad = rgb_to_lab(albedo_low,normalize=False,mode='numpy')[:,:,0]
    lab_alb_high = rgb_to_lab(albedo_high,normalize=False,mode='numpy')[:,:,0]

    # pick overlap values
    l_shad = lab_alb_shad[overlap_mask==1]
    l_high = lab_alb_high[overlap_mask==1]

    # get fit
    alb_scale = np.linalg.lstsq(l_high.reshape(-1, 1), l_shad.reshape(-1, 1), rcond=None)[0]
        
    #TODO: RANSAC

    # scale albedo
    albedo_high_scaled = albedo_high*alb_scale

    # blend
    blended_alb = alpha[:,:,np.newaxis]* albedo_high_scaled + (1-alpha[:,:,np.newaxis])*albedo_low

    return blended_alb


def blend_albedo_torch(albedo_low,albedo_high,overlap_mask,alpha):
    lab_alb_shad = rgb_to_lab(albedo_low,normalize=False,mode='torch')[:,0,:,:]
    lab_alb_high = rgb_to_lab(albedo_high,normalize=False,mode='torch')[:,0,:,:]

    # pick overlap values
    l_shad = lab_alb_shad[overlap_mask==1]
    l_high = lab_alb_high[overlap_mask==1]

    # get fit
    #alb_scale = np.linalg.lstsq(l_high.reshape(-1, 1), l_shad.reshape(-1, 1), rcond=None)[0]
    alb_scale, _, _, _ = torch.linalg.lstsq(l_high.reshape(1,-1,1), l_shad.reshape(1,-1,1), rcond=None)
        
    #TODO: RANSAC

    # scale albedo
    albedo_high_scaled = albedo_high*alb_scale.unsqueeze(3)

    # blend
    blended_alb = alpha* albedo_high_scaled + (1-alpha)*albedo_low

    return blended_alb


def separate_regions(img,thresh=0.25):

    assert thresh>=0 and thresh <= 1, 'Please use a threshold in [0..1]'
    highlight_thresh = thresh*100
    shadow_thresh = 100-highlight_thresh

    # get luminance values
    lab_img = rgb_to_lab(img,normalize=False,mode='numpy')
    l_vals = np.unique(lab_img[:,:,0])

    if l_vals.max()< highlight_thresh or l_vals.min()>shadow_thresh:
        return img, img, np.ones((img.shape[0],img.shape[1])),np.ones((img.shape[0],img.shape[1]))

    # get shadows, highlights and overlap
    high_mask = lab_img[:,:,0]>highlight_thresh
    shad_mask = lab_img[:,:,0]<shadow_thresh
    overlap_mask = high_mask*shad_mask

    # get blend mask
    alpha = lab_img[:,:,0]/100
    alpha[alpha>((1-thresh))*0.8]=(1-thresh)*0.8
    alpha[alpha<thresh*1.6]=thresh*1.6
    alpha_n = alpha-alpha.min()
    alpha_n = alpha_n/alpha_n.max()

    # scale to median luminance of 50
    l_vals_shadows = l_vals[l_vals<shadow_thresh]
    l_vals_highlights = l_vals[l_vals>highlight_thresh]

    shadow_scale = 50/np.median(l_vals_shadows)
    highlight_scale = 50/np.median(l_vals_highlights)

    shad_img_scaled = img.copy()*shadow_scale
    high_img_scaled = img.copy()*highlight_scale

    return shad_img_scaled,high_img_scaled,overlap_mask,alpha_n