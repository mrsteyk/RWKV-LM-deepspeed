import os

import torch

def load_state_dict(checkpoint, device='cpu'):
    if os.path.isdir(checkpoint):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        return get_fp32_state_dict_from_zero_checkpoint(checkpoint)
    else:
        if checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            d = load_file(checkpoint, device=device)
        else:
            d = torch.load(checkpoint, map_location='cpu')
            if list(d.keys())[0].startswith("_forward_module."):
                d = {n[len("_forward_module."):]: d[n] for n in d.keys()}
        return d

def load_checkpoint(checkpoint, model, device='cpu'):
    if os.path.isdir(checkpoint):
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        load_state_dict_from_zero_checkpoint(model, checkpoint)
    else:
        if checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            d = load_file(checkpoint, device=device)
        else:
            d = torch.load(checkpoint, map_location='cpu')
            if list(d.keys())[0].startswith("_forward_module."):
                d = {n[len("_forward_module."):]: d[n] for n in d.keys()}
        model.load_state_dict(d)
    return model
