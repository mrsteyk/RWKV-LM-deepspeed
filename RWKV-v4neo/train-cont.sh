python3 trainer.py --load_model_cont "./lightning_logs/version_8/checkpoints/epoch-epoch=02-val_loss-val_loss=1.16.ckpt/" \
--data_file train.npy --vocab_size 50278 \
--ctx_len 1024 --batch_size 2 \
--n_layer 24 --n_embd 1024 --head_qk 0 \
--lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision 16 --strategy deepspeed_stage_3_offload \
--allgather_bucket_size 100 --reduce_bucket_size 100
