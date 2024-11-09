import torch
import sys
import os
import pickle 
sys.path.append('../')


import pytorch_lightning as pl
from torch.nn import MSELoss as MSE
import numpy as np

from lit_reconstructor import LitReconstructor
from src.midas.midas_net import MidasNet_small
from src.msg_loss import MSGLoss

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


ALB_MODEL_PATH = 'checkpoints/albedo'
SH_MODEL_PATH = 'checkpoints/shading'

class LitRefiner(pl.LightningModule):

    def __init__(self,
                 lr = 1e-4,
                 img_log_frequency=50,
                 mode='effnet-inv',
                 debug=False,
                 batch_size=8, 
                 max_epochs = 100,
                 non_processed=False,
                 grad_weight = 1.0,
                 ):
        
        super().__init__()
        self.mode = mode
        self.non_processed = non_processed
        self.grad_weight = grad_weight
        
        # define architecture
        in_chan = 10
        self.refiner = MidasNet_small(activation='none',input_channels=in_chan)
        self.out_act = torch.nn.Sigmoid()

        if self.non_processed:
            # load albedo and shading reocnstruction models
            self.alb_model = LitReconstructor(
                            mode='albedo',
                            )
            # use model after training or load weights and drop into the production system
            ckpt = os.path.join(ALB_MODEL_PATH,'alb_weights.ckpt')
            self.alb_model = LitReconstructor.load_from_checkpoint(ckpt)
            self.alb_model.eval()

            self.sh_model = LitReconstructor(
                            mode='shading',
                            )
            ckpt = os.path.join(SH_MODEL_PATH,'sh_weights.ckpt')
            self.sh_model = LitReconstructor.load_from_checkpoint(ckpt)
            self.sh_model.eval()
        
        self.lr = lr
        self.n_epochs = max_epochs
        self.img_log_step = img_log_frequency

        self.eps = 1e-6
        self.batch_size = batch_size

        self.save_hyperparameters()
        self.MSG = MSGLoss()
        self.MSE = MSE(reduction='none')
        self.initialized=False
        self.debug=debug

        self.rgb_loss=[]
        self.rgb_grad_loss=[]
        self.tr_loss = []

    def initialize_aux_networks(self):
        # move reconstruction networks to device
        if self.non_processed:
            self.sh_model = self.sh_model.to(self.device)
            self.alb_model = self.alb_model.to(self.device)
        
    # dense MSE loss
    def dense_criterion(self,prediction,target,mask=None):
        if mask is None:
            mask = torch.ones_like(target)
        if mask.sum()==0:
            mask = torch.ones_like(target)
        dense_term = self.MSE(prediction, target) * mask
        dense_loss = dense_term.sum() / mask.sum()
        return dense_loss

    # multi-scale gradient loss
    def grad_criterion(self,prediction,target,mask=None):
        grad_loss = self.MSG(prediction,target,mask)
        return grad_loss

    def forward(self, x):
        x = self.refiner(x)
        x = self.out_act(x)
        return x
    

    def training_step(self, batch, batch_idx):

        # initialize aux networks if needed
        if self.initialized == False:
            self.MSG.to_device(self.device)
            self.initialize_aux_networks()
            self.initialized = True

        # get data
        rgb_ldr,rgb_gt,mask = batch['rgb_ldr'],batch['rgb'],batch['loss_mask']

        # randomly re-expose:
        prop_val = torch.rand(1)
        if prop_val<0.33:
            rgb_ldr = rgb_ldr*2**-3

        if self.non_processed:
            # run intrinsic hdr reconstruction
            alb_ldr,inv_sh_ldr = batch['alb_ldr'],batch['inv_sh_ldr']

            # albedo hallucination - expects (b,c,h,w)
            alb_mask =  torch.max(torch.clamp(rgb_ldr-0.8,0)/0.2,dim=1,keepdims=True)[0]
            alb_input_t = torch.cat([rgb_ldr, alb_ldr, alb_mask],dim=1)
            with torch.no_grad():
                alb_hdr = self.alb_model.forward(alb_input_t.float())

            
            # shading hallucination - expects (b,c,h,w)
            sh_input_t = torch.cat([rgb_ldr, inv_sh_ldr],dim=1)
            with torch.no_grad():
                inv_sh_hdr = self.sh_model.forward(sh_input_t.float())
        else:
            # load precomputed hdr components
            alb_hdr,inv_sh_hdr = batch['alb_hdr'],batch['inv_sh_hdr']

        
        # construct input
        rgb_gt = 1/(rgb_gt+1)
        
        input_t = rgb_ldr.clone()
        input_t = torch.cat([input_t,alb_hdr],dim=1)
        input_t = torch.cat([input_t,inv_sh_hdr],dim=1)
        temp_hdr = alb_hdr * (1.0/inv_sh_hdr-1.0)

        temp_hdr = 1.0/(temp_hdr+1.0)
        input_t = torch.cat([input_t,temp_hdr],dim=1)

        # run inference
        rgb_est = self.forward(input_t.float())

        # check for nans
        if rgb_est.isnan().any():
            print('Nan in rgb est')

        if rgb_gt.isnan().any():
            print('Nan in rgb gt')


        # Losses
        dense_rgb_loss = self.dense_criterion(rgb_est,rgb_gt,mask)
        msg_rgb_loss = self.grad_criterion(rgb_est,rgb_gt,mask)

        loss = dense_rgb_loss + self.grad_weight*msg_rgb_loss
        self.rgb_loss.append(dense_rgb_loss.item())
        self.rgb_grad_loss.append(msg_rgb_loss.item())
        self.tr_loss.append(loss.item())
            
        # log losses    
        self.log("Train loss", np.array(self.tr_loss).mean(),prog_bar=True,batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)
        self.log("Reconstruction loss", np.array(self.rgb_loss).mean(),batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)
        self.log("RGB grad loss",np.array(self.rgb_grad_loss).mean(),batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)

        if not self.debug:
            if batch_idx % self.img_log_step == 0:
                self.logger.log_image(key="inv rgb_gt",images=[1-rgb_gt[0]], caption=["inverse RGB GT"])
                self.logger.log_image(key="inv rgb_rec", images=[1-rgb_est[0]], caption=["inverse RGB reconstructed"])
                self.logger.log_image(key="rgb ldr", images=[rgb_ldr[0]], caption=["RGB LDR"])
                self.logger.log_image(key="albedo hdr", images=[alb_hdr[0]], caption=["Albedo HDR"])
                self.logger.log_image(key="inv sh hdr", images=[inv_sh_hdr[0]], caption=["Inverse Shading HDR"])        

        # dump input if loss contains nan
        if loss.isnan().any():
            with open(f'nan_data_{self.mode}.pkl', 'wb') as f:
                pickle.dump(batch, f)

            sys.exit(f"NaN in loss, step {batch_idx}, img {batch['path']}") 

        return loss
    

    def validation_step(self, batch, batch_idx):
        
        # initialize aux networks if needed
        if self.initialized == False:
            self.MSG.to_device(self.device)
            self.initialize_aux_networks()
            self.initialized = True

        # get data
        rgb_ldr,rgb_gt,mask = batch['rgb_ldr'],batch['rgb'],batch['loss_mask']
        
        # randomly re-expose:
        prop_val = torch.rand(1)
        if prop_val<0.33:
            rgb_ldr = rgb_ldr*2**-3

        if self.non_processed:
            # run intrinsic hdr reconstruction
            alb_ldr,inv_sh_ldr = batch['albedo'],batch['inv_shading']
            
            # albedo hallucination - expects (b,c,h,w)
            alb_mask =  torch.max(torch.clamp(rgb_ldr-0.8,0)/0.2,dim=1,keepdims=True)[0]
            alb_input_t = torch.cat([rgb_ldr, alb_ldr, alb_mask],dim=1)
            with torch.no_grad():
                alb_hdr = self.alb_model.forward(alb_input_t.float())

            # shading hallucination - expects (b,c,h,w)
            sh_input_t = torch.cat([rgb_ldr, inv_sh_ldr],dim=1)
            with torch.no_grad():
                inv_sh_hdr = self.sh_model.forward(sh_input_t.float())

        else:
            # load precomputed hdr components
            alb_hdr,inv_sh_hdr = batch['alb_hdr'],batch['inv_sh_hdr']

        # construct input
        rgb_gt = 1/(rgb_gt+1)
        input_t = rgb_ldr.clone()
        input_t = torch.cat([input_t,alb_hdr],dim=1)
        input_t = torch.cat([input_t,inv_sh_hdr],dim=1)
        
        temp_hdr = alb_hdr * (1.0/inv_sh_hdr-1.0)
        temp_hdr = 1.0/(temp_hdr+1.0)
        input_t = torch.cat([input_t,temp_hdr],dim=1)

        # run refinement inference
        rgb_est = self.forward(input_t.float())

        # check for nans
        if rgb_est.isnan().any():
            print('Nan in rgb est')

        if rgb_gt.isnan().any():
            print('Nan in rgb gt')


        # Losses
        dense_rgb_loss = self.dense_criterion(rgb_est,rgb_gt,mask)
        msg_rgb_loss = self.grad_criterion(rgb_est,rgb_gt,mask)

        loss = dense_rgb_loss + self.grad_weight*msg_rgb_loss

        # log losses
        self.log("Train loss", loss,batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)
        self.log("Reconstruction loss", dense_rgb_loss,batch_size = self.batch_size,sync_dist=True,rank_zero_only=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, betas=(0.5,0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = self.n_epochs,verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

