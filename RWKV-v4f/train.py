import argparse
import os
import time
from functools import partial
from typing import Tuple, Optional

import bitsandbytes as bnb
import deepspeed
import lightning as L

import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import numpy as np

import model as rwkv
import utils

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_dir",
        type=str,
        default="out",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default='16-mixed',
    )

    parser.add_argument(
        "--dim_att",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dim_ffn",
        type=int,
        default=0,
    )
    # parser.add_argument(
    #     "--ctx_part",
    #     type=int,
    #     default=0,
    # )
    # parser.add_argument(
    #     "--experiment_dataloader",
    #     action='store_true',
    #     default=False,
    # )
    parser.add_argument(
        "--vocab_size_delta",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--load_model_cont",
        type=str,
        default='',
    )
    parser.add_argument(
        "--soft_emb_tune",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--soft_emb_tokens",
        type=int,
        default=10,
    )

    # Original
    parser.add_argument(
        "--load_model_init",
        type=str,
        default='',
    )
    parser.add_argument(
        '--layerwise_lr',
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--ctx_len",
        type=int,
        default=1024,
    )
    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    # Hyperparameters
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--pre_ffn",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--head_qk",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--tiny_att_dim",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--tiny_att_layer",
        type=int,
        default=-999,
    )
    return parser.parse_args()

def get_batch(fabric: L.Fabric, data: np.ndarray, block_size: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    ix = torch.randint(len(data[0]) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y, None

def get_batch_mask(fabric: L.Fabric, data: np.ndarray, block_size: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    ix = torch.randint(len(data[0]) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[0][i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[0][i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    m = torch.stack([torch.from_numpy((data[1][i : i + block_size]).astype(bool)) for i in ix])
    x, y, m = fabric.to_device((x.pin_memory(), y.pin_memory(), m.pin_memory()))
    return x, y, m

def get_batch_mask_per(fabric: L.Fabric, data: np.ndarray, block_size: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    ix = torch.randint(data.shape[0], (batch_size,))
    # print(ix, data.shape)
    x = torch.stack([torch.from_numpy((data[i][0][:-1]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i][0][1:]).astype(np.int64)) for i in ix])
    m = torch.stack([torch.from_numpy((data[i][1][:-1]).astype(bool)) for i in ix])
    x, y, m = fabric.to_device((x.pin_memory(), y.pin_memory(), m.pin_memory()))
    return x, y, m

def load_datasets(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    train_data = np.memmap(os.path.join(data_dir, "train.npy"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.npy"), dtype=np.uint16, mode="r")
    return train_data, val_data

def load_datasets_per(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    train_data = np.load(os.path.join(data_dir, "train.npy"), allow_pickle=False)
    val_data = np.load(os.path.join(data_dir, "val.npy"), allow_pickle=False)
    return train_data, val_data

@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    eval_iters = 2
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets, mask = get_batch_mask_per(
            fabric,
            val_data,
            0,
            1,
        )
        logits = model(input_ids)
        if mask is not None:
            # fabric.print(logits.shape, mask.shape)
            targets = torch.where(mask, targets, -100)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

def main():
    args = parse_args()
    print(args)

    mixed = args.precision.endswith("-mixed")

    # Despite the name it is also applicable for this RNN
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={rwkv.Block})
    # strategy = L.fabric.strategies.FSDPStrategy(auto_wrap_policy=auto_wrap_policy, cpu_offload=True, accelerator="cuda", activation_checkpointing=rwkv.Block, precision=args.precision)
    # strategy = "deepspeed"
    strategy = L.fabric.strategies.DeepSpeedStrategy(
        accelerator="cuda",
        stage=3,
        pin_memory=True,
        precision=args.precision,
        offload_optimizer=True,
        offload_parameters=True,
        cpu_checkpointing=True,
        # zero_force_ds_cpu_optimizer = False,
        max_in_cpu=2_000_000_000,
        # allgather_bucket_size=100_000_000,
        # reduce_bucket_size=100_000_000,
    )

    fabric = L.Fabric(strategy=strategy, precision=args.precision)
    fabric.launch()
    fabric.seed_everything(42 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    
    train_data, val_data = load_datasets_per(".")
    
    with fabric.device:
        fabric.print("Init model")
        model = rwkv.RWKV(args)

        # блять я забыл что хотел
        # ...
        if args.load_model_cont != '' and not args.soft_emb_tune:
            # load_state_dict_from_zero_checkpoint(model, args.load_model_cont)
            pass
        elif args.load_model_init != '':
            model = load_checkpoint(args.load_model_init, model)

        if args.vocab_size_delta != 0:
            new_tokens = args.vocab_size_delta + args.vocab_size
            fabric.print(f"Resizing embed/ln with {args.vocab_size_delta} delta")
            model.resize_emb(new_tokens)
            fabric.print("Resize finished!")

        # fabric.print("Compiling model, can take a few minutes...")
        # t0 = time.time()
        # model = torch.compile(model)
        # dt = time.time() - t0
        # fabric.print(f"Compiled in {dt}!")
    
    # model = fabric.setup_module(model)

    # optimizer = bnb.optim.Adam8bit(model.configure_optimizers(), lr=args.lr_init, betas=(args.beta1, args.beta2), eps=args.adam_eps)
    # optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr_init, betas=(args.beta1, args.beta2), eps=args.adam_eps)
    # for module in model.modules():
    #     if isinstance(module, torch.nn.Embedding):
    #         fabric.print("Setting override for bnb optim Adam8bit!")
    #         bnb.optim.GlobalOptimManager.get_instance().register_module_override(module, 'weight', {'optim_bits': 32})
    # optimizer = torch.optim.AdamW(model.configure_optimizers(args), lr=args.lr_init, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=0)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_init, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=0)
    optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(model.parameters(), lr=args.lr_init, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=0, adamw_mode=False)

    # optimizer = fabric.setup_optimizers(optimizer)

    model, optimizer = fabric.setup(model, optimizer)

    # Train loop
    if args.load_model_cont != '':
        # sd = utils.load_state_dict(args.load_model_cont)
        # model.load_state_dict(sd['model'])
        # optimizer.load_state_dict(sd['optimizer'])
        sd = fabric.load(args.load_model_cont, {"model": model, "optimizer": optimizer})
        iter_num = sd['iter_num']
        save_num = sd['save_num']
    else:
        iter_num = 0
        save_num = 0
    # iter_num = 0
    # save_num = 0
    eval_interval = train_data.shape[0]
    # eval_interval = 1
    while True:
        if iter_num > 0 and iter_num % eval_interval == 0 and fabric.global_rank == 0:
            val_loss = validate(fabric, model, val_data)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            print(f"saving checkpoint to {args.out_dir}/{iter_num}")
            # torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            fabric.save(os.path.join(args.out_dir, str(iter_num)), {"model": model, "optimizer": optimizer, "iter_num": iter_num, "save_num": save_num})
            # save_num +=1

        t0 = time.time()
        
        x, y, mask = get_batch_mask_per(fabric, train_data, args.ctx_len, args.batch_size)
        logits = model(x)
        # default ignore_index is -100
        # so if we have a mask just set where mask is 0 to -100?
        if mask is not None:
            # fabric.print(logits.shape, mask.shape)
            y = torch.where(mask, y, -100)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        fabric.backward(loss)

        # or clip_val=1?
        # or max_norm=1?
        # Experimental
        # fabric.clip_gradients(model, optimizer, max_norm=1)

        optimizer.step()
        optimizer.zero_grad()

        fabric.print(f"{iter_num} {torch.mean(loss)}")

        dt = time.time() - t0
        iter_num += 1

if __name__ == "__main__":
    main()
