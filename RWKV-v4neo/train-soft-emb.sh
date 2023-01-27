python3 trainer.py --load_model_init "RWKV-4-Pile-430M-20220808-8066.pth" \
--soft_emb_tune --soft_emb_tokens 50 \
--data_file train.npy --vocab_size 50277 --vocab_size_delta 1 \
--ctx_len 1024 --batch_size 3 \
--n_layer 24 --n_embd 1024 --head_qk 0 \
--lr_init 1e-1 --lr_final 1e-1 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-6 \
--accelerator gpu --devices 1 --precision 16 --strategy deepspeed_stage_3_offload \
--allgather_bucket_size 200 --reduce_bucket_size 200