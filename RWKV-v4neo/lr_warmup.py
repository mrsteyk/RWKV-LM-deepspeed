import pytorch_lightning as pl
import math

class LearningWarmUpCallback(pl.Callback):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args

        # I use pytorch lightning checkpointing
        real_step = trainer.global_step

        # LR schedule
        w_step = args.warmup_steps
        # The only thing left is how would I count steps? Perhaps make a new argument for amount of decay steps?
        # if args.lr_final == args.lr_init or args.epoch_count == 0:
        if True:
            lr = args.lr_init
        else:
            decay_step = real_step # - args.my_pile_edecay * args.epoch_steps # I hate all the makeshift bs
            decay_total = (args.epoch_count) * args.epoch_steps
            progress = (decay_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            if args.lr_final == 0 or args.lr_init == 0:  # linear decay
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:  # exp decay
                lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))

            if trainer.global_step < w_step:
                lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)
            # if trainer.is_global_zero:
            #     print(trainer.global_step, decay_step, decay_total, w_step, progress, lr)

        for param_group in trainer.optimizers[0].param_groups:
            if args.layerwise_lr:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr
        
        pl_module.log("train_lr", lr)
