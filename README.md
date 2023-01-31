# RWKV-LM mrsteyk's """fork"""

This was originally supposed to be `RWKV-LM/RWKV-v4neo` with Microsoft's DeepSpeed oriented codebase, but it never became that and then grew up to be my ground for experiments.

# [Check out and star the original](https://github.com/BlinkDL/RWKV-LM)

# Experiments

 * **Soft Embedding fine-tuning.** This freezes all the weights and trains the additional tensor which is then concatenated with the embedding.
 * **Finetune with ZeRO Stage 3.** Original codebase doesn't work because Stage 3 shards state_dict and optim states.
 * **FP16 cuda kernel.** Don't do it, it's not worth the effort... Loss scale will skyrocket down to the abyss which can't be expressed. And it's also slower for Pascal architecture, which I am stuck with for the time being.
