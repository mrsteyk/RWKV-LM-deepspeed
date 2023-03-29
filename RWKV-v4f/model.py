import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F

import deepspeed

T_MAX = 1689 # int(os.environ["RWKV_T_MAX"])  # TAKES LOTS OF VRAM!
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load

wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"], verbose=True, extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 32) == 0
        # if "32" in os.environ["RWKV_FLOAT_MODE"]:
        #     w = -torch.exp(w.contiguous())
        #     u = u.contiguous()
        #     k = k.contiguous()
        #     v = v.contiguous()
        # else:
        dtype = k.dtype
        w = -torch.exp(w.float().contiguous())
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        ctx.save_for_backward(w, u, k, v, y)
        # if "32" in os.environ["RWKV_FLOAT_MODE"]:
        #     return y
        # elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
        #     return y.half()
        # elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
        #     return y.bfloat16()
        return y.to(dtype=dtype)
    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 32) == 0
        w, u, k, v, y = ctx.saved_tensors
        dtype = gy.dtype
        gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
        gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
        gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
        gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
        # if "32" in os.environ["RWKV_FLOAT_MODE"]:
        #     wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
        # else:
        wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        # if "32" in os.environ["RWKV_FLOAT_MODE"]:
        #     return (None, None, None, gw, gu, gk, gv)
        # elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
        #     return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        # elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
        #     return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        return (None, None, None, gw.to(dtype=dtype), gu.to(dtype=dtype), gk.to(dtype=dtype), gv.to(dtype=dtype))


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w, u, k, v)

# WKV based on my JAX experiments
def WKV_(time_decay: torch.Tensor, time_first: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    dtype = k.dtype
    device = k.device
    time_decay, time_first, k, v = time_decay.float(), time_first.float(), k.float(), v.float()
    B, T, C = k.shape
    # print(k.shape, time_decay.shape)
    time_decay = -torch.exp(time_decay)

    # B T C -> T B C aka 0 1 2 -> 1 0 2
    #k, v = k.transpose(1, 0, 2), v.transpose(1, 0, 2)
    k = k.swapaxes(1, 0)
    v = v.swapaxes(1, 0)

    aa = torch.zeros((B, C), device=device)
    bb = torch.zeros((B, C), device=device)
    pp = torch.full((B, C), -torch.inf, device=device)

    # gc.collect()
    # torch.cuda.empty_cache()

    a, b = None, None
    for t in range(T):
        ww = time_first + k[t]
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)

        # PyTorch should be lazy when it comes to computing tensor vals?
        # if t == (T - 1):
        a = e1 * aa + e2 * v[t]
        b = e1 * bb + e2

        ww = pp + time_decay
        p = torch.maximum(ww, k[t])
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        aa = e1 * aa + e2 * v[t]
        bb = e1 * bb + e2
        pp = p

    # print(a.shape, b.shape, k.shape)
    return (aa, bb, pp), (a / b).to(dtype=dtype).swapaxes(1, 0)

# WKV = torch.compile(WKV_)

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

class RWKV_TimeMix(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            
            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for h in range(args.dim_att):
                decay_speed[h] = -5 + 8 * (h / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(args.dim_att)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(args.dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

    # @MyFunction
    # @torch.compile
    def jit_func(self, x):
        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    # @torch.compile
    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)
        sr, k, v = self.jit_func(x)
        rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)
        # rwkv = sr * WKV(self.time_decay, self.time_first, k, v)[1]
        return self.output(rwkv)

########################################################################################################

class RWKV_ChannelMix(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    # @MyFunction
    # @torch.compile
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

class MishGLU(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    # @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            self.att = RWKV_TimeMix(args, layer_id)

        self.ffn = RWKV_ChannelMix(args, layer_id)
        
        # if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
        #     self.tiny_ln = nn.LayerNorm(args.n_embd)
        #     self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
        #     self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
        #     self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
        #     self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)

        if self.layer_id == 0 and args.pre_ffn > 0:
            x = x + self.ffnPre(self.ln1(x))
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att') or args.dim_att == 0:
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn') or args.dim_ffn == 0:
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, 'tiny_att_layer'):
            args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim'):
            args.tiny_att_dim = -1

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
        
    def resize_emb(self, new_tokens: int):
        # rank_zero_info(f"### RESIZING MODEL TO {new_tokens} TOKENS ###")
        
        new_embed = nn.Embedding(new_tokens, self.args.n_embd)
        new_embed.to(self.emb.weight.device, dtype=self.emb.weight.dtype)
        # nn.init.zeros_(new_embed.weight)
        # Should be a BIT better than full zeroes???
        nn.init.orthogonal_(new_embed.weight)
        
        n = min(self.args.vocab_size, new_tokens)
        
        # rank_zero_info("### Start emb copy %s %s", new_embed.weight.size(), self.emb.weight.size())
        new_embed.weight.data[:n, :] = self.emb.weight.data[:n, :]
        self.emb = new_embed
        # rank_zero_info("### emb copy end")

        # Now we resize head
        new_head = nn.Linear(self.args.n_embd, new_tokens, bias=False)
        new_head.to(self.head.weight.device, dtype=self.head.weight.dtype)
        nn.init.orthogonal_(new_head.weight, gain=1 * 0.5)

        # rank_zero_info("### Start head copy %s %s", new_head.weight.size(), self.head.weight.size())
        new_head.weight.data[:n, :] = self.head.weight.data[:n, :]
        self.head = new_head
        # rank_zero_info("### RESIZE END")
    
    def configure_optimizers(self, args):
        # args = self.args
        if args.layerwise_lr:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if "time_mix" in n:
                    lr_1x.add(n)
                elif "time_decay" in n:
                    lr_2x.add(n)
                elif "time_first" in n:
                    lr_3x.add(n)
                else:
                    lr_1x.add(n)
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))
            # print('1x', lr_1x)
            # print('2x', lr_2x)
            # print('3x', lr_3x)
            param_dict = {n: p for n, p in self.named_parameters()}
            optim_groups = [
                {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
            ]
        else:
            optim_groups = [
                {"params": [p for n, p in self.named_parameters()], "weight_decay": 0.0},
            ]

        return optim_groups

    def forward(self, idx):
        # args = self.args
        B, T = idx.size()
        # We are full PyTorch here!
        # assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        x_emb = x

        # Checkpointing is handled by FDSP by Meta FairSeq
        # if args.tiny_att_dim > 0:
        #     for block in self.blocks:
        #         x = block(x, x_emb)
        # else:
        for block in self.blocks:
            # x = block(x)
            x = deepspeed.checkpointing.checkpoint(block, x)

        x = self.ln_out(x)

        # if args.head_qk > 0:
        #     q = self.head_q(x)[:, :T, :]
        #     k = self.head_k(x)[:, :T, :]
        #     c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
        #     c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

        #     # if "32" in os.environ["RWKV_FLOAT_MODE"]:
        #     #     c = c @ F.one_hot(idx, num_classes=args.vocab_size)
        #     # elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
        #     #     c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
        #     # elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
        #     #     c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()

        #     x = self.head(x) + c
        # else:
        x = self.head(x)

        return x
