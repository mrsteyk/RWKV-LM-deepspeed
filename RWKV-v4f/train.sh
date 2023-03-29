#!/bin/sh
python3 train.py --data_file train.npy --vocab_size 50277 --vocab_size_delta 2 --ctx_len 1024 --batch_size 1 --n_layer 24 --n_embd 1024 --head_qk 0 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8
