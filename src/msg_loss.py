import torchvision.transforms.functional as TF
import kornia.morphology as kn_morph
import torch

from src.image_derivative import ImageDerivative

class MSGLoss():
    def __init__(self,scales=4,taps=[1,1,1,1], k_size=[3,3,3,3], device=None, eps = 0.00001):
        self.n_scale = scales
        self.taps = taps
        self.k_size = k_size
        self.device = device
        self.eps = eps
        
        assert len(self.taps) == self.n_scale, 'number of scales and number of taps must be the same'
        assert len(self.k_size) == self.n_scale, 'number of scales and number of kernels must be the same'

        self.imgDerivative = ImageDerivative()

        self.erod_kernels = []
        for tap in self.taps:
            kernel = torch.ones(2*tap+1, 2*tap+1)
            self.erod_kernels.append(kernel)

        if self.device is not None:
            self.to_device(self.device)

    def to_device(self, device):
        self.imgDerivative.to_device(device)
        self.device = device
        self.erod_kernels = [kernel.to(device) for kernel in self.erod_kernels]
    
    def __call__(self, output, target, mask=None):
        return self.forward(output, target, mask)

    def forward(self, output, target, mask):
        
        diff = output - target
        
        if mask is None:
            mask = torch.ones(diff.shape[0],1, diff.shape[2], diff.shape[3])
            mask = mask.to(self.device)

        loss = 0
        for i in range(self.n_scale):
            # resize with antialiase
            mask_resized = torch.floor(self.resize_aa(mask, i)+0.001)
            # erosion to mask out pixels that are effected by unkowns
            mask_eroded = kn_morph.erosion(mask_resized, self.erod_kernels[i])
            
            diff_resized = self.resize_aa(diff, i)
            
            # imshow(mask_resized, 'mask_resized')
            # imshow(diff_resized, 'diff_resized')
            # compute grads
            grad_mag = self.gradient_mag(diff_resized, i)
            
            # imshow(grad_mag, 'grad_mag')
            # mean over channels
            grad_mag_mean = torch.mean(grad_mag, dim=1, keepdim=True) 
            # imshow(grad_mag, 'grad_mag')

            # average the per pixel diffs
            temp = mask_eroded * grad_mag_mean
            # imshow(temp, 'grad_mag')

            loss += torch.sum(temp) / (torch.sum(mask_eroded) * grad_mag.shape[1]+self.eps)
        
        loss /= self.n_scale
        return loss
    
    def resize_aa(self,img, scale):
            if scale == 0:
                return img
            blurred = TF.gaussian_blur(img, self.k_size[scale])
            scaled = blurred[:, :, ::2**scale, ::2**scale]
            #blurred = img
            #scaled = torch.nn.functional.interpolate(blurred, scale_factor=1/(2**scale),mode='bilinear', align_corners=True, antialias=True)
            return scaled
    
    def gradient_mag(self, diff, scale):
        # B*C*H*W
        grad_x, grad_y = self.imgDerivative(diff, self.taps[scale])

        # B*C*H*W
        grad_magnitude = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + self.eps)
        grad_magnitude = torch.nan_to_num(grad_magnitude)

        return grad_magnitude