import argparse
import deepspeed
import os
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

import dataset
import lr_warmup
from soft_embedding_hotswap import SoftEmbedding

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_dataloader",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--experiment_fp16",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--vocab_size_delta",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--allgather_bucket_size",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--reduce_bucket_size",
        type=int,
        default=200,
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

    parser = Trainer.add_argparse_args(parser)

    return parser

from utils import load_checkpoint

if __name__ == "__main__":
    args = get_argparser().parse_args()
    args.betas = (args.beta1, args.beta2)
    rank_zero_info(args)
    
    # assert args.precision != "64"
    assert args.precision in [32, 16, "bf16"]
    if args.precision == 16:
        os.environ["RWKV_FLOAT_MODE"] = "fp16"
    else:
        os.environ["RWKV_FLOAT_MODE"] = str(args.precision)
    os.environ["RWKV_T_MAX"] = str(args.ctx_len + (args.soft_emb_tokens if args.soft_emb_tune else 0))

    os.environ["RWKVK_CUDA_FP16"] = "1" if args.experiment_fp16 else "0"

    # Now we can import the model after setting that stupid T max envvar
    import model as M
    model = M.RWKV(args)
    # model = None

    if args.load_model_cont != '' and not args.soft_emb_tune:
        # load_state_dict_from_zero_checkpoint(model, args.load_model_cont)
        pass
    elif args.load_model_init != '':
        model = load_checkpoint(args.load_model_init, model)
    else:
        # TODO?
        # model = M.RWKV(args)
        model.generate_init_weight()
    
    if args.precision == 16:
        model = model.half()
    elif args.precision == "bf16":
        model = model.bfloat16()
    else:
        model = model.float()

    if args.vocab_size_delta > 0:
        new_vocab_size = args.vocab_size + args.vocab_size_delta
        model.resize_emb(new_vocab_size)
        args.vocab_size = new_vocab_size
    
    if args.soft_emb_tune:
        # meme hard, die young
        print("### буду погибать молодым/малоДЫМ(а)")
        args.layerwise_lr = False
        for p in model.parameters():
            p.requires_grad = False
        model.emb_hotswap = True
        assert args.soft_emb_tokens < args.vocab_size, "Soft Embedding can't eat more than the `emb`"
        model.emb = SoftEmbedding(model.emb, n_tokens=args.soft_emb_tokens, initialize_from_vocab=True)

    lr_meme = lr_warmup.LearningWarmUpCallback(args)
    device_stats = DeviceStatsMonitor(cpu_stats=True)
    val_loss_checkpointing = ModelCheckpoint(
        filename="epoch-{epoch:02d}-step-{step:03d}-val_loss-{val_loss:.2f}",
        # save_on_train_epoch_end=True,
        # save_weights_only=True,
        save_top_k=3,
        mode='min',
        monitor="val_loss",
        auto_insert_metric_name=False,
        every_n_train_steps=None if not args.experiment_dataloader else 300,
    )
    epoch_checkpointing = ModelCheckpoint(
        filename="epoch-{epoch:02d}-step-{step:03d}",
        save_on_train_epoch_end=True,
        save_top_k=1,
        mode='max',
        monitor="epoch",
        auto_insert_metric_name=False,
        every_n_train_steps=None if not args.experiment_dataloader else 300,
    )

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[lr_meme, device_stats, val_loss_checkpointing, epoch_checkpointing],
    )
    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.allgather_bucket_size * 1e6
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.reduce_bucket_size * 1e6
    rank_zero_info(trainer.strategy.config)

    train_data = dataset.MyDataSet(args)

    # TODO(mrsteyk): Allow different validation files
    # use 20% of training data for validation
    train_set_size = int(len(train_data) * 0.8)
    valid_set_size = len(train_data) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_set_size, valid_set_size], generator=seed)

    # data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.batch_size, num_workers=1, persistent_workers=False, drop_last=True)
    train_loader = DataLoader(train_data, shuffle=True, pin_memory=True, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_data, shuffle=False, pin_memory=True, batch_size=args.batch_size)

    trainer.fit(model, train_loader, valid_loader, ckpt_path=args.load_model_cont if args.load_model_cont != ''  else None)
