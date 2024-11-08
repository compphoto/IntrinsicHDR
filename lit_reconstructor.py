import torch
import sys
import os
import cv2
import pickle 
sys.path.append('../')

import pytorch_lightning as pl
import torchvision.transforms.functional as TF

from torch.nn import MSELoss as MSE

import kornia as kn
import numpy as np

from src.midas.midas_net import MidasNet_small
from src.msg_loss import MSGLoss
from intrinsic_decomposition.common.model_util import load_models
from intrinsic_decomposition.common.general import round_32

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class LitReconstructor(pl.LightningModule):

    def __init__(self,
                 lr = 1e-4,
                 img_log_frequency=50,
                 mode='combined',
                 debug=False,
                 debug_dir = 'checkpoints/debug',
                 ord_path='',
                 mrg_path='',
                 batch_size=8, 
                 rgb_weight = 1.0, 
                 alb_weight=1.0,
                 sh_weight=1.0,
                 rgb_grad_weight = 1.0, 
                 alb_grad_weight=1.0,
                 sh_grad_weight=1.0,
                 max_epochs = 100,
                 scale_mode='input',
                 preprocessed=True,
                 backbone='unet',
                 ):
        
        super().__init__()
        self.mode = mode
        self.backbone = backbone
        self.out_act = torch.nn.Sigmoid()

        if mode == 'albedo':
            self.iteration = self.albedo_iteration
            self.in_channels = 7
            self.out_channels = 3
        elif mode == 'shading':
            self.iteration = self.shading_iteration
            self.in_channels = 4
            self.out_channels = 1
        else:
            self.iteration = self.rgb_iteration
            self.in_channels = 7
            self.out_channels = 3
        
        self.network = MidasNet_small(activation='none',input_channels=self.in_channels,output_channels=self.out_channels)


        self.rgb_weight = rgb_weight
        self.alb_weight = alb_weight
        self.sh_weight = sh_weight

        self.rgb_grad_weight = rgb_grad_weight
        self.alb_grad_weight = alb_grad_weight
        self.sh_grad_weight = sh_grad_weight

        self.lr = lr
        self.n_epochs = max_epochs
        self.img_log_step = img_log_frequency
        self.ord_path = ord_path
        self.mrg_path = mrg_path
        self.eps = 1e-6
        self.batch_size = batch_size

        self.scale_mode = scale_mode

        self.save_hyperparameters()
        self.MSG = MSGLoss()
        self.MSE = MSE(reduction='none')
        self.initialized=False
        self.debug=debug
        self.debug_dir = debug_dir
        self.preprocessed = preprocessed

        self.rgb_loss=[]
        self.rgb_grad_loss=[]
        self.alb_loss=[]
        self.alb_grad_loss = []
        self.sh_loss = []
        self.sh_grad_loss=[]
        self.tr_loss = []

    # initialize the decomposition models
    def initialize_aux_networks(self):
        if not self.preprocessed:
            self.intrinsic_models = load_models(self.ord_path,self.mrg_path,device=self.device)
    
    # decomposition util: brightness computation
    def get_brightness(self,rgb):
        brightness = (0.3 * rgb[:,0,:,:]) + (0.59 * rgb[:,1,:,:]) + (0.11 * rgb[:,2, :,:])
        return brightness.unsqueeze(1)

    # decomposition util: double estimation
    def equalize_predictions(self, img, base, full, p=0.5):

        b,c,h, w = img.shape

        full_shd = (1. / full.clamp(1e-5)) - 1.
        base_shd = (1. / base.clamp(1e-5)) - 1.

        full_alb = self.get_brightness(img) / full_shd.clamp(1e-5)
        base_alb = self.get_brightness(img) / base_shd.clamp(1e-5)

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
        new_full_shd = self.get_brightness(img) / new_full_alb.clamp(1e-5)
        new_full = 1.0 / (1.0 + new_full_shd)

        return base, new_full,success

    # decomposition util: decomposition
    def decompose(self, img_arr,resize_conf=None,base_size=384,lstsq_p=0.0):
        models = self.intrinsic_models
        _,_,orig_h, orig_w = img_arr.shape
        
        if resize_conf == None:
            img_arr = TF.resize(img_arr, (round_32(orig_h), round_32(orig_w)), antialias=True)

        else:
            scale = resize_conf / max(orig_h, orig_w)
            img_arr = TF.resize(img_arr, (round_32(orig_h * scale), round_32(orig_w * scale)), antialias=True)

        _,_,fh, fw = img_arr.shape
        
        lin_img = img_arr
        
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
            
            ord_base, ord_full,success = self.equalize_predictions(lin_img, base_out, full_out, p=lstsq_p)
            if success==False:
                return None,None,success

            combined = torch.cat((lin_img, ord_base, ord_full),dim=1)
            inv_shd = models['real_model'](combined.float()).unsqueeze(1)
            
            shd = ((1.0 / inv_shd) - 1.0)
            alb = lin_img / shd
            # ------------------------------------------------------
                      
        inv_shd = TF.resize(inv_shd, (orig_h, orig_w), antialias=True)
        alb = TF.resize(alb, (orig_h, orig_w), antialias=True)
        
        return inv_shd,alb,success

  
    # masked MSE loss
    def dense_criterion(self,prediction,target,mask=None):
        if mask is None:
            mask = torch.ones_like(target)
        if mask.sum()==0:
            mask = torch.ones_like(target)
        dense_term = self.MSE(prediction, target) * mask
        dense_loss = dense_term.sum() / mask.sum()
        return dense_loss

    # masked Multi-Scale Gradient Loss
    def grad_criterion(self,prediction,target,mask=None):
        grad_loss = self.MSG(prediction,target,mask)
        return grad_loss

    def forward(self, x):
        x = self.network(x)
        x = self.out_act(x)
        return x
    
    # fit scale between albedo estimation and ground truth
    def get_albedo_scale(self,rgb,alb_ldr,alb_hdr,mask):
        b,_,_,_ = rgb.shape
        ldr_scale = 0.9/torch.quantile((alb_ldr*mask).reshape(b,-1),0.9,dim=1)
        ldr_scale = torch.minimum(torch.ones((b,1,1,1)).to(alb_ldr.device),ldr_scale.reshape(b,1,1,1))
        alb_ldr = alb_ldr*ldr_scale
        hdr_scale = torch.clamp(torch.linalg.lstsq((alb_hdr*mask).reshape(b,-1,1),(alb_ldr*mask).reshape(b,-1,1))[0],0,None)
        ldr_scale = ldr_scale.reshape(b,1,1,1)
        hdr_scale = hdr_scale.reshape(b,1,1,1)

        return ldr_scale,hdr_scale

    
    def albedo_iteration(self,batch,batch_idx):
        albedo_gt,inv_shading_gt, rgb_gt,rgb_ldr,loss_mask = batch['albedo'],batch['inv_shading'],batch['rgb'],batch['rgb_ldr'],batch['loss_mask']
        b,_,_,_ = albedo_gt.shape

        if self.preprocessed:
            albedo_ldr, inv_shading_ldr = batch['alb_ldr'],batch['inv_sh_ldr']
            success=True
        else:
            # Intrinsic decomposition
            inv_shading_ldr,albedo_ldr,success = self.decompose(rgb_ldr.float())
            if success == False:
                return rgb_ldr,inv_shading_gt,albedo_gt,rgb_gt,inv_shading_gt,albedo_gt,loss_mask,success
        

        # match albedo scale to ground truth due to decomposition ambiguity - Suppl. Sec. E
        ldr_scale,hdr_scale = self.get_albedo_scale(rgb_ldr,albedo_ldr,albedo_gt,loss_mask)

        albedo_ldr_scaled = albedo_ldr*ldr_scale
        albedo_gt_scaled = albedo_gt*hdr_scale
        shading_ldr = (1/inv_shading_ldr)-1
        shading_gt = (1/inv_shading_gt)-1
        shading_ldr_scaled = shading_ldr/ldr_scale
        shading_gt_scaled = shading_gt/hdr_scale

        # normalize gt to [0..1]
        if albedo_gt_scaled.max()>1:
            max_scale = 1/torch.quantile((albedo_gt_scaled*loss_mask).reshape(b,-1),0.99,dim=1)
            max_scale = torch.minimum(torch.ones((b,1,1,1)).to(albedo_ldr_scaled.device),max_scale.reshape(b,1,1,1))
            albedo_ldr_scaled = albedo_ldr_scaled * max_scale
            albedo_gt_scaled = albedo_gt_scaled * max_scale
            shading_gt_scaled = shading_gt_scaled/max_scale
            shading_ldr_scaled = shading_ldr_scaled/max_scale

        # convert to inverse representation - Sec. 
        inv_shading_ldr_scaled = 1/(shading_ldr_scaled+1)
        inv_shading_gt_scaled = 1/(shading_gt_scaled+1)

        # construct guidance mask indicating highlight regions - Sec.
        guide_mask = torch.nan_to_num(torch.max(torch.clamp(rgb_ldr-0.8,0)/0.2,dim=1,keepdims=True)[0],nan=0,posinf=1,neginf=0)

        # re-expose:
        prop_val = torch.rand(1)
        if prop_val<0.33:
            rgb_ldr = rgb_ldr*2**-3

        # Inference
        input_data = torch.cat([rgb_ldr,albedo_ldr_scaled,guide_mask],dim =1)
        albedo_est = self.forward(input_data.float())

        # Reconstruction
        rgb_est = albedo_est * shading_gt_scaled
        shading_est = self.get_brightness(rgb_gt/torch.clamp(albedo_est,1e-6))
        inv_shading_est = 1/(shading_est+1)
        return rgb_est,inv_shading_est,albedo_est,albedo_ldr_scaled,inv_shading_ldr_scaled,rgb_gt,inv_shading_gt_scaled,albedo_gt_scaled,loss_mask,success
                                                                                        

    def shading_iteration(self,batch,batch_idx):
        albedo_gt,inv_shading_gt, rgb_gt,rgb_ldr,loss_mask = batch['albedo'],batch['inv_shading'],batch['rgb'],batch['rgb_ldr'],batch['loss_mask']
        b,_,_,_ = albedo_gt.shape

        if self.preprocessed:
            albedo_ldr, inv_shading_ldr = batch['alb_ldr'],batch['inv_sh_ldr']
            success=True
        else:
            # Intrinsic decomposition
            inv_shading_ldr,albedo_ldr,success = self.decompose(rgb_ldr.float())
            if success == False:
                return rgb_ldr,inv_shading_gt,albedo_gt,rgb_gt,inv_shading_gt,albedo_gt,loss_mask,success
        
        # match albedo scale to ground truth due to decomposition ambiguity - Suppl. Sec. E
        ldr_scale,hdr_scale = self.get_albedo_scale(rgb_ldr,albedo_ldr,albedo_gt,loss_mask)

        albedo_ldr_scaled = albedo_ldr*ldr_scale
        albedo_gt_scaled = albedo_gt*hdr_scale
        shading_ldr = (1/inv_shading_ldr)-1
        shading_gt = (1/inv_shading_gt)-1
        shading_ldr_scaled = shading_ldr/ldr_scale
        shading_gt_scaled = shading_gt/hdr_scale

        # normalize gt to [0..1]
        if albedo_gt_scaled.max()>1:
            max_scale = 1/torch.maximum((albedo_gt_scaled*loss_mask).reshape(b,-1).max(dim=1)[0],(albedo_ldr_scaled*loss_mask).reshape(b,-1).max(dim=1)[0])
            max_scale = torch.minimum(torch.ones((b,1,1,1)).to(albedo_ldr_scaled.device),max_scale.reshape(b,1,1,1))
            albedo_ldr_scaled = albedo_ldr_scaled * max_scale
            albedo_gt_scaled = albedo_gt_scaled * max_scale
            shading_gt_scaled = shading_gt_scaled/max_scale
            shading_ldr_scaled = shading_ldr_scaled/max_scale

        # convert to inverse representation - Sec.
        inv_shading_ldr = 1/(torch.clamp(shading_ldr_scaled,1e-6)+1)
        inv_shading_gt = 1/(torch.clamp(shading_gt_scaled,1e-6)+1)

        # re-expose:
        prop_val = torch.rand(1)
        if prop_val<0.33:
            rgb_ldr = rgb_ldr*2**-3

        # Inference
        input_data = torch.cat([rgb_ldr,inv_shading_ldr],dim =1)
        inv_shading_est  = self.forward(input_data.float())

        # Reconstruction
        shading_est = 1/torch.clamp(inv_shading_est,1e-6)-1
        rgb_est = albedo_gt_scaled*shading_est
        albedo_est = rgb_gt/torch.clamp(shading_est,1e-6)
        return rgb_est,inv_shading_est,albedo_est,albedo_ldr_scaled,inv_shading_ldr,rgb_gt,inv_shading_gt,albedo_gt_scaled,loss_mask,success


    def training_step(self, batch, batch_idx):
        # initialize decomposition networks
        if self.initialized == False:
            self.MSG.to_device(self.device)
            self.initialize_aux_networks()
            self.initialized = True

        # run inference
        rgb_est,inv_shading_est,albedo_est,albedo_ldr,inv_shading_ldr,rgb_gt,inv_shading_gt,albedo_gt,mask,success = self.iteration(batch,batch_idx)
        
        if success == False:
            # least-square solution is not found
            print('Skip image')
            return None

        # Debugging
        if rgb_est.isnan().any():
            print('Nan in rgb est')
        if inv_shading_est.isnan().any():
            print('Nan in inv_shading est')
        if albedo_est.isnan().any():
            print('Nan in alb est')
        if rgb_gt.isnan().any():
            print('Nan in rgb gt')
        if inv_shading_gt.isnan().any():
            print('Nan in inv_shading gt')
        if albedo_gt.isnan().any():
            print('Nan in alb gt')

        # invert
        rgb_est = 1/(rgb_est+1)
        rgb_gt = 1/(rgb_gt+1)

        # Losses
        dense_rgb_loss = self.dense_criterion(rgb_est,rgb_gt,mask)
        msg_rgb_loss = self.grad_criterion(rgb_est,rgb_gt,mask)

        loss = self.rgb_weight*dense_rgb_loss + self.rgb_grad_weight*msg_rgb_loss
        self.rgb_loss.append(dense_rgb_loss.item())
        self.rgb_grad_loss.append(msg_rgb_loss.item())
    
        # auxillary losses
        dense_sh_loss = self.dense_criterion(inv_shading_est,inv_shading_gt,mask)
        msg_sh_loss = self.grad_criterion(inv_shading_est,inv_shading_gt,mask)

        dense_alb_loss = self.dense_criterion(albedo_est,albedo_gt,mask)
        msg_alb_loss = self.grad_criterion(albedo_est,albedo_gt,mask)

        loss += self.sh_weight*dense_sh_loss + self.sh_grad_weight*msg_sh_loss + self.alb_weight*dense_alb_loss + self.alb_grad_weight*msg_alb_loss

        self.tr_loss.append(loss.item())
        self.alb_loss.append(dense_alb_loss.item())
        self.sh_loss.append(dense_sh_loss.item())
        self.alb_grad_loss.append(msg_alb_loss.item())
        self.sh_grad_loss.append(msg_sh_loss.item())
        
        # logging
        self.log("Train loss", np.array(self.tr_loss).mean(),prog_bar=True,batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)
        self.log("Reconstruction loss", np.array(self.rgb_loss).mean(),batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)
        self.log("RGB grad loss",np.array(self.rgb_grad_loss).mean(),batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)
        self.log("Shading loss", np.array(self.sh_loss).mean(),batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)
        self.log("SH grad loss", np.array(self.sh_grad_loss).mean(),batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)
        self.log("Albedo loss", np.array(self.alb_loss).mean(),batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)
        self.log("Alb grad loss", np.array(self.alb_grad_loss).mean(),batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)

        if not self.debug:
            if batch_idx % self.img_log_step == 0:   
                self.logger.log_image(key="inv rgb_gt",images=[1-rgb_gt[0]], caption=[f"{batch['path'][0]}: inverse RGB GT"])
                self.logger.log_image(key="inv rgb_rec", images=[1-rgb_est[0]], caption=["inverse RGB reconstructed"])
                self.logger.log_image(key="shading_est", images=[inv_shading_est[0]], caption=["Predicted Inverse Shading"])
                self.logger.log_image(key="albedo_est", images=[albedo_est[0]], caption=["Predicted Albedo"])
                self.logger.log_image(key="shading_gt", images=[inv_shading_gt[0]], caption=["Inverse Shading GT"])
                self.logger.log_image(key="albedo_gt", images=[albedo_gt[0]], caption=["Albedo GT"])
                self.logger.log_image(key="shading_ldr", images=[inv_shading_ldr[0]], caption=["inverse Shading Input"])
                self.logger.log_image(key="albedo_ldr", images=[albedo_ldr[0]], caption=["Albedo Input"])            

        # output full batches to debug directory
        else:
            if batch_idx % self.img_log_step == 0:
                 
                for b in range(len(batch)):
                    print(rgb_gt[b].min(),rgb_gt[b].max())
                    b_rgb_est = (1-rgb_est[b]).squeeze().permute(1,2,0).detach().cpu().numpy() 
                    b_rgb_est = cv2.cvtColor(b_rgb_est,cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(self.debug_dir,f'rgb_est_{b}.png'),np.uint16(b_rgb_est*66536.0))

                    b_inv_sh_est = inv_shading_est[b].squeeze().detach().cpu().numpy() 
                    cv2.imwrite(os.path.join(self.debug_dir,f'inv_sh_est_{b}.png'),np.uint16(b_inv_sh_est*66536.0))

                    b_albedo_est = albedo_est[b].permute(1,2,0).detach().cpu().numpy()
                    b_albedo_est = cv2.cvtColor(b_albedo_est,cv2.COLOR_RGB2BGR) 
                    cv2.imwrite(os.path.join(self.debug_dir,f'albedo_est_{b}.png'),np.uint16(b_albedo_est*66536.0))

                    b_rgb_ldr = batch['rgb_ldr'][b].permute(1,2,0).detach().cpu().numpy()
                    b_rgb_ldr = cv2.cvtColor(b_rgb_ldr,cv2.COLOR_RGB2BGR) 
                    cv2.imwrite(os.path.join(self.debug_dir,f'rgb_ldr_{b}.png'),np.uint16(b_rgb_ldr*66536.0))

                    b_inv_shading_ldr = inv_shading_ldr[b].permute(1,2,0).detach().cpu().numpy()
                    cv2.imwrite(os.path.join(self.debug_dir,f'inv_shading_ldr_{b}.png'),np.uint16(b_inv_shading_ldr*66536.0))

                    b_albedo_ldr = albedo_ldr[b].permute(1,2,0).detach().cpu().numpy()
                    b_albedo_ldr = cv2.cvtColor(b_albedo_ldr,cv2.COLOR_RGB2BGR) 
                    cv2.imwrite(os.path.join(self.debug_dir,f'albedo_ldr_{b}.png'),np.uint16(b_albedo_ldr*66536.0))


                    b_rgb_gt = (1-rgb_gt[b]).permute(1,2,0).detach().cpu().numpy()
                    b_rgb_gt = cv2.cvtColor(b_rgb_gt,cv2.COLOR_RGB2BGR) 
                    cv2.imwrite(os.path.join(self.debug_dir,f'rgb_gt_{b}.png'),np.uint16(b_rgb_gt*65536.0))

                    b_inv_shading_gt = inv_shading_gt[b].permute(1,2,0).detach().cpu().numpy()
                    cv2.imwrite(os.path.join(self.debug_dir,f'inv_shading_gt_{b}.png'),np.uint16(b_inv_shading_gt*65536.0))

                    b_albedo_gt = albedo_gt[b].permute(1,2,0).detach().cpu().numpy()
                    b_albedo_gt = cv2.cvtColor(b_albedo_gt,cv2.COLOR_RGB2BGR) 
                    cv2.imwrite(os.path.join(self.debug_dir,f'albedo_gt_{b}.png'),np.uint16(b_albedo_gt*66536.0))
                        

        # dump input if loss contains nan
        if loss.isnan().any():
            with open(f'nan_data_{self.mode}.pkl', 'wb') as f:
                pickle.dump(batch, f)

            sys.exit(f"NaN in loss, step {batch_idx}, img {batch['path']}") 

        return loss
    
    def validation_step(self, batch, batch_idx):
        # initialize decomposition networks
        if self.initialized == False:
            self.MSG.to_device(self.device)
            self.initialize_aux_networks()
            self.initialized = True

        # run inference
        rgb_est,inv_shading_est,albedo_est,_,_,rgb_gt,inv_shading_gt,albedo_gt,mask,success = self.iteration(batch,batch_idx)

        if success == False:
            # least-square solution is not found, ignore batch
            print('Skip image')
            return None
        
        # invert
        rgb_est = 1/(rgb_est+1)
        rgb_gt = 1/(rgb_gt+1)


        # Losses
        dense_rgb_loss = self.dense_criterion(rgb_est,rgb_gt,mask)
        msg_rgb_loss = self.grad_criterion(rgb_est,rgb_gt,mask)

        loss = self.rgb_weight*dense_rgb_loss + self.rgb_grad_weight*msg_rgb_loss

     
        # auxillary losses
        dense_sh_loss = self.dense_criterion(inv_shading_est,inv_shading_gt,mask)
        msg_sh_loss = self.grad_criterion(inv_shading_est,inv_shading_gt,mask)

        dense_alb_loss = self.dense_criterion(albedo_est,albedo_gt,mask)
        msg_alb_loss = self.grad_criterion(albedo_est,albedo_gt,mask)

        loss += self.sh_weight*dense_sh_loss + self.sh_grad_weight*msg_sh_loss + self.alb_weight*dense_alb_loss + self.alb_grad_weight*msg_alb_loss

        # logging
        self.log("Train loss", loss,batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)
        self.log("Reconstruction loss", dense_rgb_loss,batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)
        self.log("Shading loss", dense_sh_loss,batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)
        self.log("Albedo loss", dense_alb_loss,batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, betas=(0.5,0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = self.n_epochs,verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


