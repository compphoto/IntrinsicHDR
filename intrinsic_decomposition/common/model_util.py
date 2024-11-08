import sys
from os.path import dirname, abspath
import torch
from pathlib import Path

parent_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_path)

from intrinsic_decomposition.networks.altered_midas.midas_net import MidasNet
from intrinsic_decomposition.networks.altered_midas.midas_net_custom import MidasNet_small

def load_models(
    ord_path='boosted_shading/results/final_weights/vivid_bird_318_300.pt',
    mrg_path='boosted_shading/results/final_weights/fluent_eon_138_200.pt',
    device='cuda'
):

    models = {}

    ord_model = MidasNet()
    ord_model.load_state_dict(torch.load(ord_path))
    ord_model.eval()
    ord_model = ord_model.to(device)
    
    mrg_model = MidasNet_small(exportable=False, input_channels=5, output_channels=1)
    mrg_model.load_state_dict(torch.load(mrg_path)) 
    mrg_model.eval()
    mrg_model = mrg_model.to(device)

    models['ordinal_model'] = ord_model
    models['real_model'] = mrg_model
    
    return models

