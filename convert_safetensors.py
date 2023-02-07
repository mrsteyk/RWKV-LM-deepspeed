import argparse
import os

import torch
from safetensors.torch import save_file, load_file

def convert_rnn(w, float_mode="fp32", rescale_layer=6):
    from torch.nn import functional as F
    # RWKV_RESCALE_LAYER = 6
    # refine weights and send to correct device
    keys = list(w.keys())
    if 'pos_emb_x' in keys:
        w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']).reshape(args.ctx_len+1, -1)[:-1,:]
    keys = list(w.keys())
    for x in keys:
        block_id = 0
        if 'blocks.' in x:
            block_id = int(x.split('.')[1])
        if 'att.output.weight' in x:
            w[x] = w[x] / (2 ** int(block_id // rescale_layer))
        if 'ffn.value.weight' in x:
            w[x] = w[x] / (2 ** int(block_id // rescale_layer))
                        
        if '.time_' in x:
            w[x] = w[x].squeeze()
        if '.time_decay' in x:
            w[x] = w[x].float()
            w[x] = -torch.exp(w[x])
        elif '.time_first' in x:
            w[x] = w[x].float()
        else:
            if float_mode == "fp32":
                w[x] = w[x].float()
            elif float_mode == "bf16":
                w[x] = w[x].bfloat16()
            elif float_mode == "fp16":
                w[x] = w[x].half()

        w[x].requires_grad = False
    
    w["emb.weight"] = F.layer_norm(w["emb.weight"], (w["emb.weight"].shape[1],), weight=w["blocks.0.ln0.weight"], bias=w["blocks.0.ln0.bias"])
    return w

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_checkpoint", type=str)
    parser.add_argument("--output", type=str, required=False)
    parser.add_argument("--zero_tag", type=str, required=False)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--bf16", action='store_true')
    parser.add_argument("--rnn", action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if os.path.isdir(args.input_checkpoint):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        d = get_fp32_state_dict_from_zero_checkpoint(args.input_checkpoint, tag=args.zero_tag)
    else:
        if args.input_checkpoint.endswith(".safetensors"):
            d = load_file(args.input_checkpoint)
        else:
            d = torch.load(args.input_checkpoint, map_location='cpu')
    if list(d.keys())[0].startswith("_forward_module."):
        d = {n[len("_forward_module."):]: d[n] for n in d.keys()}
    if args.output is None:
        args.output = f"{os.path.splitext(args.input_checkpoint)[0]}.safetensors"
    elif not args.output.endswith("safetensors"):
        args.output = f"{args.output}s" if args.output.endswith("safetensor") else f"{args.output}.safetensors"
    
    if not args.rnn:
        if args.fp16:
            d = {k: v.half() for k, v in d.items()}
        elif args.bf16:
            d = {k: v.bfloat16() for k, v in d.items()}
        else:
            d = {k: v.float() for k, v in d.items()}
    else:
        d = convert_rnn(d, float_mode="fp16" if args.fp16 else "bf16" if args.bf16 else "fp32")

    save_file(d, args.output)
