import argparse
import deepspeed
import os
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint

import dataset
import lr_warmup

def get_argparser():
    parser = argparse.ArgumentParser()

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
    os.environ["RWKV_T_MAX"] = str(args.ctx_len)

    # Now we can import the model after setting that stupid T max envvar
    import model as M
    model = M.RWKV(args)
    # model = None

    if args.load_model_init != '':
        d = torch.load(args.load_model_init, map_location='cpu')
        model.load_state_dict(d)
        # model = M.RWKV(args).load_from_checkpoint(args.load_model_init)
    else:
        # TODO?
        # model = M.RWKV(args)
        model.generate_init_weight()

    if args.vocab_size_delta > 0:
        new_vocab_size = args.vocab_size + args.vocab_size_delta
        model.resize_emb(new_vocab_size)
        args.vocab_size = new_vocab_size

    lr_meme = lr_warmup.LearningWarmUpCallback(args)
    device_stats = DeviceStatsMonitor(cpu_stats=True)
    val_loss_checkpointing = ModelCheckpoint(
        filename="epoch-{epoch:02d}-val_loss-{val_loss:.2f}",
        # save_on_train_epoch_end=True,
        # save_weights_only=True,
        save_top_k=3,
        mode='min',
        monitor="val_loss",
    )
    epoch_checkpointing = ModelCheckpoint(
        filename="epoch-{epoch:02d}",
        save_on_train_epoch_end=True,
        save_top_k=2,
        mode='max',
        monitor="epoch",
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

    trainer.fit(model, train_loader, valid_loader)