# RWKV-LM mrsteyk's """fork"""

This was originally supposed to be `RWKV-LM/RWKV-v4neo` with Microsoft's DeepSpeed oriented codebase, but it never became that and then grew up to be my ground for experiments.

# [Check out and star the original](https://github.com/BlinkDL/RWKV-LM)

# Experiments

 * **Soft Embedding fine-tuning.** This freezes all the weights and trains the additional tensor which is then concatenated with the embedding.
 * **Finetune with ZeRO Stage 3.** Original codebase doesn't work because Stage 3 shards state_dict and optim states.
 * **FP16 cuda kernel.** Don't do it, it's not worth the effort... Loss scale will skyrocket down to the abyss which can't be expressed. And it's also slower for Pascal architecture, which I am stuck with for the time being.
 * **More GPT-esque dataloader.** This uses a per token basis for lookup. Limits how much actual data validation sees (full split for me is 40 hours to check) if you use limit_val_batch, but for the training (aka finetuning, I don't experiment with pretraining yet) I think it should work better because model sees more random points in the "middle" and learns how to continue that. (read as more variety in the training data for the entropy)
 * **RNN code "optimisation" and soft embed.** RWKV-v4neo doesn't use pos_embed (at least all the models I tested) so we don't ever need to keep the full context in memory. As for the soft embedding - slight code refactor of separating into `forward` and `forward_raw`, later takes the raw embedding tensor.
 * **[Alpha Sampling](https://platform.openai.com/docs/api-reference/parameter-details)** aka `mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence` #nuffsaid
