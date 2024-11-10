import sys
from os.path import dirname, abspath
import torch
from pathlib import Path

parent_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_path)

from intrinsic_decomposition.networks.altered_midas.midas_net import MidasNet
from intrinsic_decomposition.networks.altered_midas.midas_net_custom import MidasNet_small

def load_models(
    ord_path='vivid_bird_318_300.pt',
    mrg_path='fluent_eon_138_200.pt',
    device='cuda'
):

    models = {}
    base_url = 'https://github.com/compphoto/IntrinsicHDR/releases/download/v1.0/'

    ## uncomment for offline work
    # base_url = './intrinsic_decomposition/pretrained_weights/'

    ord_model = MidasNet()

    ## comment next two lines for offline work
    ord_state_dict = torch.hub.load_state_dict_from_url(base_url + ord_path , map_location=device, progress=True)
    ord_model.load_state_dict(ord_state_dict)

    ## uncomment for offline work
    # ord_model.load_state_dict(torch.load(ord_path))

    ord_model.eval()
    ord_model = ord_model.to(device)
    ord_model.device = device
    
    mrg_model = MidasNet_small(exportable=False, input_channels=5, output_channels=1)

    ## comment next two lines for offline work
    mrg_state_dict = torch.hub.load_state_dict_from_url(base_url + mrg_path , map_location=device, progress=True)
    mrg_model.load_state_dict(mrg_state_dict)

    ## uncomment for offline work
    # mrg_model.load_state_dict(torch.load(mrg_path)) 
    mrg_model.eval()
    mrg_model = mrg_model.to(device)
    mrg_model.device = device

    models['ordinal_model'] = ord_model
    models['real_model'] = mrg_model
    
    return models

