# 8190/8 or 4096/4
# too little vram, sadge
python3 trainer.py \
--data_file train_pre.npy --experiment_dataloader --vocab_size 100288 \
--ctx_len 2048 --ctx_part 2 --batch_size 1 \
--n_layer 12 --n_embd 512 --head_qk 0 --dim_att 1024 --dim_ffn 1024 \
--lr_init 8e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision 16 --strategy deepspeed_stage_3_offload \
--allgather_bucket_size 100 --reduce_bucket_size 100 \
--log_every_n_steps 5 --default_root_dir /mnt/@home/mrsteyk/rwkv/ --val_check_interval 100 --limit_val_batches 300
