import argparse
import os

import torch
from safetensors.torch import save_file

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_checkpoint", type=str)
    parser.add_argument("--output", type=str, required=False)
    parser.add_argument("--zero_tag", type=str, required=False)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--bf16", action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if os.path.isdir(args.input_checkpoint):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        d = get_fp32_state_dict_from_zero_checkpoint(args.load_model_init, tag=args.zero_tag)
    else:
        d = torch.load(args.input_checkpoint, map_location='cpu')
    if list(d.keys())[0].startswith("_forward_module."):
        d = {n[len("_forward_module."):]: d[n] for n in d.keys()}
    if args.output is None:
        args.output = f"{os.path.splitext(args.input_checkpoint)[0]}.safetensors"
    elif not args.output.endswith("safetensors"):
        args.output = f"{args.output}s" if args.output.endswith("safetensor") else f"{args.output}.safetensors"
    
    if args.fp16:
        d = {k: v.half() for k, v in d.items()}
    elif args.bf16:
        d = {k: v.bfloat16() for k, v in d.items()}

    save_file(d, args.output)
