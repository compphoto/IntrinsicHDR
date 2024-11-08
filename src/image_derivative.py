import kornia.filters as kn_filters
import torch

class ImageDerivative():
    def __init__(self, device=None):
        # seprable kernel: first derivative, second prefiltering
        tap_3 = torch.tensor([[0.425287, -0.0000, -0.425287], [0.229879, 0.540242, 0.229879]])
        tap_5 = torch.tensor([[0.109604,  0.276691,  0.000000, -0.276691, -0.109604], [0.037659,  0.249153,  0.426375,  0.249153,  0.037659]])
        tap_7 = torch.tensor([[0.018708,  0.125376,  0.193091,  0.000000, -0.193091, -0.125376, -0.018708],[0.004711,  0.069321,  0.245410,  0.361117,  0.245410,  0.069321,  0.004711]])
        tap_9 = torch.tensor([[0.0032, 0.0350, 0.1190, 0.1458, -0.0000, -0.1458, -0.1190, -0.0350, -0.0032], [0.0009, 0.0151, 0.0890, 0.2349, 0.3201, 0.2349, 0.0890, 0.0151, 0.0009]])
        tap_11 = torch.tensor([0])
        tap_13 = torch.tensor([[0.0001, 0.0019, 0.0142, 0.0509, 0.0963, 0.0878, 0.0000, -0.0878, -0.0963, -0.0509, -0.0142, -0.0019, -0.0001],
                                [0.0000, 0.0007, 0.0071, 0.0374, 0.1126, 0.2119, 0.2605, 0.2119, 0.1126, 0.0374, 0.0071, 0.0007, 0.0000]])
        self.kernels=[tap_3, tap_5, tap_7, tap_9, tap_11, tap_13]

        # sending them to device
        if device is not None:
            self.to_device(device)

    def to_device(self,device):
        self.kernels = [kernel.to(device) for kernel in self.kernels]

    def __call__(self, img, t_id):
        # 
        # img : B*C*H*W
        # t_id : tap radius [for example t_id=1 will use the tap 3] 
        
        if t_id == 5:
            assert False, "Not Implemented"
        return self.forward(img, t_id)
    
    def forward(self, img, t_id=1):
        kernel = self.kernels[t_id-1]
        
        p = kernel[1:2,...]
        d1 = kernel[0:1,...]
        
        # B*C*H*W
        grad_x = kn_filters.filter2d_separable(img, p, d1, border_type='reflect', normalized=False, padding='same')
        grad_y = kn_filters.filter2d_separable(img, d1, p, border_type='reflect', normalized=False, padding='same')

        return (grad_x,grad_y)
    