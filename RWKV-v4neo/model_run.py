########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types
import torch
import math, os, gc
from torch.nn import functional as F
import torch.nn as nn
from typing import List, Dict

MyModule = nn.Module
def __nop(ob):
    return ob
MyFunction = __nop

# # try torchdynamo
# import torchdynamo
# MyFunction = torchdynamo.optimize(os.environ["RWKV_RUN_BACKEND"]) # !!!BUGGY!!! wrong output

# try torch jit --> faster for fp32, slower for fp16 (why?)
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

RWKV_HEAD_QK_DIM = 0
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM} RWKV_JIT_ON {os.environ["RWKV_JIT_ON"]}\n')

DEBUG_TIME = False   # True False - show trained time-coeffs

RWKV_RESCALE_LAYER = 6 # set x=x/2 every X layer

############################################################################################################

from utils import load_state_dict

class RWKV_RNN(MyModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.FLOAT_MODE = args.FLOAT_MODE
        self.RUN_DEVICE = args.RUN_DEVICE

        with torch.no_grad():
            w = load_state_dict(args.load_model_init) #torch.load(args.load_model_init, map_location='cpu')
            # refine weights and send to correct device
            keys = list(w.keys())
            if 'pos_emb_x' in keys:
                w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']).reshape(args.ctx_len+1, -1)[:-1,:]
            keys = list(w.keys())
            print_need_newline = False
            for x in keys:
                block_id = 0
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                if 'att.output.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                if 'ffn.value.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                                
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                    if DEBUG_TIME:
                        print(x, w[x].numpy())
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                elif '.time_first' in x:
                    w[x] = w[x].float()
                else:
                    if self.FLOAT_MODE == "fp32":
                        w[x] = w[x].float()
                    elif self.FLOAT_MODE == "bf16":
                        w[x] = w[x].bfloat16()
                    elif self.FLOAT_MODE == "fp16":
                        w[x] = w[x].half()

                w[x].requires_grad = False
                if args.RUN_DEVICE == 'cuda':
                    w[x] = w[x].cuda()

                if x == 'emb.weight' or 'ln0' in x:
                    continue

                if block_id == 0:
                    if print_need_newline:
                        print('\n', end = '')
                        print_need_newline = False
                    print(x.ljust(35), str(w[x].dtype).replace('torch.', '').ljust(10), w[x].device)
                else:
                    print_need_newline = True
                    print('.', end = '', flush = True)

        # store weights in self.w
        keys = list(w.keys())
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        with torch.no_grad(): # precompute embedding
            x = self.LN(self.w.emb.weight, self.w.blocks[0].ln0)
            self.w.emb.weight = x.to(dtype=self.dtype)

        self.eval()
        gc.collect()
        torch.cuda.empty_cache()
    
    @property
    def dtype(self):
        return torch.float32 if self.FLOAT_MODE == "fp32" else torch.float16 if self.FLOAT_MODE == "fp16" else torch.bfloat16

    def LN(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    @MyFunction
    def FF(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xx = state[5*i+0].type(self.dtype)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+0] = x.float()

        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))
        kv = vw @ k

        return r * kv

    @MyFunction
    def SA(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xx = state[5*i+1].type(self.dtype)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+1] = x.float()

        r = torch.sigmoid(rw @ xr)
        k = (kw @ xk).float()
        v = (vw @ xv).float()

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        ww = pp + time_decay
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        wkv = (a / b).type(self.dtype)
        
        return ow @ (r * wkv)

    def forward_raw(self, x, state, preprocess_only = False):
        with torch.no_grad():
            w = self.w
            args = self.args

            if state == None:
                state = torch.zeros(args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)
                for i in range(args.n_layer):
                    state[5*i+4] -= 1e30

            for i in range(args.n_layer):
                # if i == 0:
                #     x = self.LN(x, w.blocks[i].ln0)
                
                ww = w.blocks[i].att
                x = x + self.SA(self.LN(x, w.blocks[i].ln1), state, i, 
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                
                ww = w.blocks[i].ffn
                x = x + self.FF(self.LN(x, w.blocks[i].ln2), state, i, 
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight)
                
                if (i+1) % RWKV_RESCALE_LAYER == 0:
                    x = x / 2

            if preprocess_only:
                return state

            x = self.LN(x, w.ln_out)
            x = w.head.weight @ x

            return x.float(), state

    def forward_optimised(self, token, state, preprocess_only = False):
        # Optimisation is that pos_embed DOES NOT FUCKING EXIST SO WE DONT NEED TO KEEP ALL CONTEXT!!!
        with torch.no_grad():
            assert not hasattr(self.w, "pos_emb")
            x = self.w.emb.weight[token]
            if self.RUN_DEVICE == 'cuda':
                x = x.cuda()
            return self.forward_raw(x, state, preprocess_only=preprocess_only)

    def forward(self, ctx, state, preprocess_only = False):
        with torch.no_grad():
            assert not hasattr(self.w, "pos_emb")
            x = self.w.emb.weight[ctx[-1]]
            if self.RUN_DEVICE == 'cuda':
                x = x.cuda()
            return self.forward_raw(x, state, preprocess_only=preprocess_only)
    
    def forward_ln0(self, x, state, preprocess_only = False):
        with torch.no_grad():
           self.forward_raw(self.LN(x, self.w.blocks[0].ln0), state, preprocess_only=preprocess_only)
