import torch

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import cv2
import numpy as np
from tqdm import tqdm

from intrinsic_decomposition.common.model_util import load_models
from intrinsic_decomposition.common.general import round_32
from lit_reconstructor import LitReconstructor
from lit_refiner import LitRefiner

from src.color_utils import rgb_to_lab
from src.decomposition_utils import decompose_torch, get_quantile

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


def blend_imgs(ldr,hdr,mask):
    """
    Blends two images based on a mask

    Args:
    ldr: np.array, ldr image
    hdr: np.array, hdr image
    mask: np.array, mask image

    Returns:
    blended: np.array, blended image
    """

    # convert to lab
    lab_ldr = rgb_to_lab(ldr,normalize=False,mode='numpy')[:,:,0]
    lab_hdr = rgb_to_lab(hdr,normalize=False,mode='numpy')[:,:,0]

    # pick overlap values
    l_ldr = lab_ldr[mask>=0]
    l_hdr = lab_hdr[mask>=0]

    # get fit
    scale = np.linalg.lstsq(l_ldr.reshape(-1, 1), l_hdr.reshape(-1, 1), rcond=None)[0]
    
    # scale ldr
    ldr_scaled = ldr*scale

    # blend
    blended = mask[:,:,np.newaxis]* hdr + (1-mask[:,:,np.newaxis])*ldr_scaled

    return blended



def load_reconstruction_models(device,model_root = 'https://github.com/compphoto/IntrinsicHDR/releases/download/v1.0/'):
    """
    Load reconstruction models

    Args:
    model_root: str, project root, default='.'

    Returns:
    sh_model: torch model, shading model
    alb_model: torch model, albedo model
    ref_model: torch model, refinement model
    """

    # ------------
    # shading model
    # ------------
    sh_model = LitReconstructor(
                        mode='shading',
                        )

    ## uncomment for offline working
    # ckpt = os.path.join(model_root,'checkpoints/shading','sh_weights.ckpt')
    
    ## comment for offline working
    ckpt = model_root + 'sh_weights.ckpt'

    sh_model = LitReconstructor.load_from_checkpoint(ckpt)
    sh_model.to(device)
    sh_model.eval()
    print('Shading model loaded ...')    


    # ------------
    # albedo model
    # ------------
    alb_model = LitReconstructor(
                        mode='albedo',
                        )
    ## uncomment for offline working
    # ckpt = os.path.join(model_root,'checkpoints/albedo','alb_weights.ckpt')

    ## comment for offline working
    ckpt = model_root + 'alb_weights.ckpt' 
    alb_model = LitReconstructor.load_from_checkpoint(ckpt)
    alb_model.to(device)
    alb_model.eval()
    print('Albedo model loaded ...')

    # ------------
    # refinement model
    # ------------
    ref_model = LitRefiner()

    ## uncomment for offline working
    # ckpt = os.path.join(model_root,'checkpoints/refinement','ref_weights.ckpt') 
    # checkpoint = torch.load(ckpt)

    ## comment for offline working
    checkpoint = torch.hub.load_state_dict_from_url(model_root + 'ref_weights.ckpt', progress=True)

    # ignore potential albedo and shading weights
    refiner_weights = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("refiner.")}
    ref_model.load_state_dict(refiner_weights)
    ref_model.to(device)
    ref_model.eval()
    print('Refinement model loaded ...')

    return sh_model,alb_model,ref_model


def hdr_reconstruction(reconstruction_networks,albedo_raw,inv_shading_raw,ldr_t,proc_scale=1.0):
    """
    Reconstruct HDR image from intrinsic components

    Args:
    reconstruction_networks: tuple, reconstruction networks
    albedo_raw: torch.tensor, albedo tensor
    shading_raw: torch.tensor, shading tensor
    ldr_t: torch.tensor, ldr tensor
    proc_scale: float, processing scale

    Returns:
    hdr_r: np.array, hdr image
    """

    ldr_t = ldr_t.to(albedo_raw.device)

    # get guide 
    mask =  torch.max(torch.clamp(ldr_t-0.8,0,1)/0.2,dim=1,keepdims=True)[0]

    
    # Scale albedo:
    # due to the scale ambiguity of the decomposition, the scale
    # of the predicted albedo can vary greatly between images.
    # We scale the albedo to have a 95% quantile of 0.95 evenly for all images.
    alb_scale = 0.95/get_quantile(albedo_raw,0.95) #
    albedo = albedo_raw * alb_scale            
    sh = 1.0/inv_shading_raw - 1.0
    inv_shading = 1/(sh/alb_scale +1.0)

    # albedo hallucination - expects (b,c,h,w)
    alb_model = reconstruction_networks[1]
    alb_input_t = torch.cat([torch.clamp(ldr_t*proc_scale,0,1), albedo, mask],dim=1)
    with torch.no_grad():
        albedo_hdr = alb_model.forward(alb_input_t.float().to(alb_model.device))
    
    # shading hallucination - expects (b,c,h,w)
    sh_model = reconstruction_networks[0]
    sh_input_t = torch.cat([torch.clamp(ldr_t*proc_scale,0,1), inv_shading],dim=1)
    with torch.no_grad():
        inv_sh_hdr = sh_model.forward(sh_input_t.float().to(sh_model.device))


    # refinement - expects (b,c,h,w)
    ref_model = reconstruction_networks[2]
    shading_hdr = (1.0/inv_sh_hdr -1.0)
    hdr_t = albedo_hdr * shading_hdr
    inv_hdr_t = 1.0/(hdr_t+1.0)
    input_t = torch.cat([torch.clamp(ldr_t.to(ref_model.device)*proc_scale,0,1),inv_hdr_t,albedo_hdr,inv_sh_hdr],dim=1)
    with torch.no_grad():    
        ref_hdr = ref_model.forward(input_t.float())

    ref_hdr = (1.0/ref_hdr)-1.0


    # output
    rgb_hdr = ref_hdr.squeeze().permute(1,2,0).cpu().numpy()
    albedo_hdr = albedo_hdr.squeeze().permute(1,2,0).cpu().numpy()
    inv_sh_hdr = inv_sh_hdr.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    albedo = albedo.squeeze().permute(1,2,0).cpu().numpy()
    inv_shading = inv_shading.squeeze().cpu().numpy()
    

    return rgb_hdr, albedo_hdr, inv_sh_hdr, albedo, inv_shading, mask


def intrinsic_hdr(decomp_models, 
                  reconstruction_networks, 
                  ldr_c, 
                  max_res=4096,
                  decomp_res=None, 
                  proc_scale=1.0):
    """
    Intrinsic HDR processing

    Args:
    decomp_models: tuple, decomposition models
    reconstruction_networks: tuple, reconstruction networks
    ldr_c: np.array, ldr image
    max_res: int, maximum resolution
    proc_scale: float, processing scale

    Returns:
    results: dict, intrinsic hdr results
    """

    # norm
    ldr_c = np.clip(ldr_c,0,1)
    
    # secure resize
    h_in,w_in = ldr_c.shape[:2]
    if max(h_in,w_in)>max_res:
        s = max_res/max(h_in,w_in)
        h_proc = h_in * s
        w_proc = w_in * s
    else:
        h_proc = h_in
        w_proc = w_in

    # resize to closest multiple of 32 for decomposition
    new_h, new_w = round_32(h_proc), round_32(w_proc)
    ldr_lin = cv2.resize(ldr_c,(new_w,new_h))
        
    # convert to torch
    ldr_t = torch.tensor(ldr_lin*proc_scale).permute(2,0,1).unsqueeze(0)

    # intrinsic decomposition
    pred_inv_shading_raw,pred_albedo_raw = decompose_torch(decomp_models,torch.clamp(ldr_t,0,1), decomp_res)

    # reconstruct and refine
    rec_results = hdr_reconstruction(reconstruction_networks,pred_albedo_raw,pred_inv_shading_raw,ldr_t,proc_scale)


    # resize to original resolution
    hdr_r = cv2.resize(rec_results[0],(w_in,h_in))

    # blend new highlights onto original image
    bl_mask = cv2.resize(rec_results[5],(w_in,h_in))
    hdr_r = blend_imgs(ldr_c,hdr_r,bl_mask)

    # resize intrinsic components
    alb_hdr = cv2.resize(rec_results[1],(w_in,h_in))
    shading_hdr = (1.0/rec_results[2]-1.0)
    shading_hdr = cv2.resize(shading_hdr,(w_in,h_in))

    alb_raw = pred_albedo_raw.squeeze().permute(1,2,0).cpu().numpy()
    alb_raw = cv2.resize(alb_raw,(w_in,h_in))

    sh_raw = (1.0/pred_inv_shading_raw-1.0).squeeze().cpu().numpy()
    sh_raw = cv2.resize(sh_raw,(w_in,h_in))

    alb_ldr = cv2.resize(rec_results[3],(w_in,h_in))
    sh_ldr = (1.0/rec_results[4]-1.0)
    sh_ldr = cv2.resize(sh_ldr,(w_in,h_in))

    # pack results
    results = {
        'rgb_hdr':hdr_r,
        'alb_hdr':alb_hdr,
        'sh_hdr':shading_hdr,
        'mask':bl_mask,
        'alb_raw':alb_raw,
        'sh_raw':sh_raw,
        'alb_ldr':alb_ldr,
        'sh_ldr':sh_ldr,
    }

    return results



if __name__=='__main__':

    # ------------
    # args
    # ------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_imgs', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=None)

    parser.add_argument('--res', type=int, default=None, help='Processing resolution.')
    parser.add_argument('--img_scale', type=float,default=1.0)

    parser.add_argument('--use_exr',action="store_true")
    parser.add_argument('--testing',action="store_true")
    parser.add_argument('--subfolder_structure',action="store_true")
    parser.add_argument('--testset',action="store_true")
    
    args = parser.parse_args()


    # ------------
    # decomposition models
    # ------------
    decomp_models = load_models(
            ord_path='vivid_bird_318_300.pt',
            mrg_path='fluent_eon_138_200.pt',
            device = DEVICE
        )
    print('Decomposition models loaded ...')


    # ------------
    # reconstruction models
    # ------------

    # uncomment for offline working and pass to load_reconstruction_models
    # model_root = '.'

    reconstruction_models = load_reconstruction_models(DEVICE)
    print('Reconstruction models loaded ...')


    # ------------
    # data
    # ------------
    # get images
    if args.subfolder_structure:
        # keep subfolder structure
        imgs = sorted(glob.glob(os.path.join(args.test_imgs,'**','*.exr')))[args.start_id:args.end_id]
    else:
        imgs = sorted(glob.glob(os.path.join(args.test_imgs, '*.exr')))[args.start_id:args.end_id]


    # create output folder
    run_name = 'predictions'
    out_path = os.path.join(args.output_path,run_name)
    os.makedirs(out_path,exist_ok=True)

    # create subfolder for refined images
    ref_out_path = out_path 
    os.makedirs(ref_out_path,exist_ok=True)

    # define output file type
    if args.use_exr:
        ext = '.exr'
    else:
        ext = '.hdr'


    # ------------
    # inference
    # ------------
    for img_name in tqdm(imgs):
        fpath,fname = os.path.split(img_name)
        print(f'Processing img {fname} ...')

        # input
        ldr_in = cv2.imread(img_name,cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        ldr_c = (cv2.cvtColor(ldr_in,cv2.COLOR_BGR2RGB)).astype(np.float32)

        # run intrinsic hdr reconstruction
        results = intrinsic_hdr(decomp_models, reconstruction_models, ldr_c)

        # unpack results
        hdr_r = results['rgb_hdr']

        # save refined hdr image
        if args.subfolder_structure:
            ref_img_out_path = fpath.replace(args.test_imgs,ref_out_path+'/')
            os.makedirs(ref_img_out_path,exist_ok=True)
            ref_hdr_path = os.path.join(ref_img_out_path,fname.replace('.exr',ext))
        else:
            ref_hdr_path = os.path.join(ref_out_path+'/',fname.replace('.exr',ext))
        cv2.imwrite(ref_hdr_path,cv2.cvtColor(hdr_r,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_EXR_COMPRESSION,1])


    print("Finished!")