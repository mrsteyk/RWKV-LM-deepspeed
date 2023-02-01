# !!! MAKESHIFT INFERENCE SCRIPT !!!

import argparse
import copy
import gc
import os
import torch
import numpy as np

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint, get_fp32_state_dict_from_zero_checkpoint

from soft_embedding_hotswap import SoftEmbedding

from transformers import PreTrainedTokenizerFast

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--greedy",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--experiment_rnn",
        action='store_true',
        default=False,
    )

    # Stuff from the trainer
    parser.add_argument(
        "--accelerator",
        type=str,
        default='gpu'
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16",
    )

    parser.add_argument(
        "--vocab_size_delta",
        type=int,
        default=0,
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
    parser.add_argument(
        "--soft_emb_checkpoint",
        type=str,
        default='',
    )

    # Original
    parser.add_argument(
        "--load_model_init",
        type=str,
        default='',
    )
    parser.add_argument(
        "--ctx_len",
        type=int,
        default=1024,
    )
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

def inference_rnn(args):
    os.environ["RWKV_JIT_ON"] = "1" if args.precision != 16 else "0"
    args.RUN_DEVICE = "cpu" if args.accelerator != "gpu" else "cuda"
    args.FLOAT_MODE = "fp32" if args.precision == "32" else "fp16" if args.precision == 16 else "bf16"
    os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE

    print(args.RUN_DEVICE, args.FLOAT_MODE, args.precision)

    import model_run as M
    model = M.RWKV_RNN(args)
    # model.freeze()

    if args.vocab_size_delta > 0:
        print("Я ебал всё в рот", args.vocab_size_delta, "раз")
        new_vocab_size = args.vocab_size + args.vocab_size_delta
        
        # model.resize_emb(new_vocab_size)
        print("До АДшек", model.w.emb.weight.shape)
        model.w.emb.weight = torch.cat([model.w.emb.weight, torch.zeros((args.vocab_size_delta, args.n_embd), dtype=model.w.emb.weight.dtype, device=model.w.emb.weight.device)])
        print("После АДшек", model.w.emb.weight.shape)

        print("До духовки", model.w.head.weight.shape)
        new_head = torch.nn.Linear(args.n_embd, new_vocab_size, bias=False)
        torch.nn.init.orthogonal_(new_head.weight, gain=1 * 0.5)
        new_head.to(model.w.head.weight.device, dtype=model.w.head.weight.dtype)
        new_head.weight.data[:args.vocab_size, :] = model.w.head.weight.data[:args.vocab_size, :]
        model.w.head.weight.data = new_head.weight.data
        print("После духовки", model.w.head.weight.shape)

        args.vocab_size = new_vocab_size

    gc.collect()
    torch.cuda.empty_cache()

    print("\n!!! EXPERIMENTAL INFERENCE WITH RNN !!!")

    state = None
    if args.soft_emb_tune and args.soft_emb_checkpoint != '':
        if os.path.isdir(args.soft_emb_checkpoint):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(args.soft_emb_checkpoint)
        else:
            state_dict = torch.load(args.soft_emb_checkpoint, map_location='cpu')
        print("Soft Emb keys", state_dict.keys())
        #model.emb.learned_embedding = state_dict["_forward_module.emb.learned_embedding"]
        # model.load_state_dict({"emb.learned_embedding": state_dict["_forward_module.emb.learned_embedding"]}, strict=False)
        soft_emb = state_dict[list(state_dict.keys())[0]]
        soft_emb._requires_grad = False
        if args.precision == 16:
            soft_emb = soft_emb.half()
        elif args.precision == "bf16":
            soft_emb = soft_emb.bfloat16()
        if args.accelerator == "gpu":
            soft_emb = soft_emb.cuda()
        print("Start preprocessing state for RNN with soft emb...", soft_emb)

        for i in range(soft_emb.shape[0]):
            state = model.forward_raw(soft_emb[i], state, preprocess_only=True)
        
        print("Finished preprocess of soft embed state!")
    
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = PreTrainedTokenizerFast(tokenizer_file='../20B_tokenizer_openchatgpt.json')
    context = """quality: high

[System]
Assistant is a distilled language model trained by the community.<|STK_SP|>

[System]
<|STK_SP|>

[User]
Can you explain quantum computing?<|STK_SP|>

[Assistant]
"""

    print("### Preprocess initial context!")
    ctx_src = tokenizer.encode(context)
    for i in ctx_src[:-1]:
        state = model.forward_optimised(i, state, preprocess_only=True)
    logits_src, state_src = model.forward(ctx_src, state, preprocess_only=False)
    print("### Finished preprocessing initial context!")

    MIN_LEN = 100
    EOS = 0
    END = 50277
    TRIALS = 1 if args.greedy else 3
    for _ in range(TRIALS):
        print("--- --- ---")
        # ctx = copy.deepcopy(ctx_src)
        logits = logits_src.clone()
        state = copy.deepcopy(state_src)

        gc.collect()
        torch.cuda.empty_cache()

        for i in range(767):
            if not args.greedy:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                # print(probs, probs.shape)
                
                sorted_probs = torch.sort(probs, descending=True)[0]
                # print("sorted", sorted_probs, sorted_probs.shape)

                cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
                cutoff = float(sorted_probs[np.argmax(cumulative_probs > 0.99)])
                probs[probs < cutoff] = 0
                # print("cut probs", probs, probs.shape)
                if True:
                    probs[EOS] = 0
                if i < MIN_LEN:
                    probs[EOS] = 0
                    probs[END] = 0
                
                try:
                    out = torch.multinomial(probs.float(), num_samples=1)[0]
                except:
                    out = EOS
                # print(out, out.shape)
            else:
                out = torch.argmax(logits)
            
            if out == EOS:
                print("<|BAD|>", end='')
            if out == EOS or out == END:
                break
            # ctx += [int(out)]
            print(tokenizer.decode(out), end='', flush=True)
            # break

            # Really dumb way ig
            logits, state = model.forward_optimised(int(out), state, preprocess_only=False)
        print()


if __name__ == "__main__":
    args = get_args()
    print(args)

    assert args.precision in ["32", "16", "bf16"]
    if args.precision == "16":
        args.precision = 16
        os.environ["RWKV_FLOAT_MODE"] = "fp16"
    else:
        os.environ["RWKV_FLOAT_MODE"] = str(args.precision)
    os.environ["RWKV_T_MAX"] = str(args.ctx_len + (args.soft_emb_tokens if args.soft_emb_tune else 0))
    os.environ["RWKVK_CUDA_FP16"] = "0"

    if args.experiment_rnn:
        inference_rnn(args)
        os._exit(0)

    # Now we can import the model after setting that stupid T max envvar
    import model as M
    model = M.RWKV(args)
    # model = None

    if args.load_model_init != '':
        if os.path.isdir(args.load_model_init):
            load_state_dict_from_zero_checkpoint(model, args.load_model_init)
            model = model.cpu()
        else:
            d = torch.load(args.load_model_init, map_location='cpu')
            if list(d.keys())[0].startswith("_forward_module."):
                d = {n[len("_forward_module."):]: d[n] for n in d.keys()}
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
    
    if args.soft_emb_tune:
        # meme hard, die young
        print("### буду погибать молодым/малоДЫМ(а)")
        for p in model.parameters():
            p.requires_grad = False
        model.emb_hotswap = True
        assert args.soft_emb_tokens < args.vocab_size, "Soft Embedding can't eat more than the `emb`"
        model.emb = SoftEmbedding(model.emb, n_tokens=args.soft_emb_tokens, initialize_from_vocab=True)

        if args.soft_emb_checkpoint != '':
            if os.path.isdir(args.soft_emb_checkpoint):
                state_dict = get_fp32_state_dict_from_zero_checkpoint(args.soft_emb_checkpoint)
                model = model.cpu()
            else:
                state_dict = torch.load(args.soft_emb_checkpoint, map_location='cpu')
            print("Soft Emb keys", state_dict.keys())
            #model.emb.learned_embedding = state_dict["_forward_module.emb.learned_embedding"]
            model.load_state_dict({"emb.learned_embedding": state_dict["_forward_module.emb.learned_embedding"]}, strict=False)
    
    if args.precision == 16:
        model = model.half()
    elif args.precision == "bf16":
        model = model.bfloat16()
    
    assert args.accelerator in ["gpu", "cpu"]
    if args.accelerator == "gpu":
        model = model.cuda()
    elif args.accelerator == "cpu":
        model = model.cpu()
    
    print("!!! WARNING: THIS IS A MAKESHIFT INFERENCE SCRIPT! !!!")
    model.freeze()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='../20B_tokenizer_openchatgpt.json')
    context = """quality: high

[System]
Assistant is a distilled language model trained by the community.<|STK_SP|>

[System]
<|STK_SP|>

[User]
Can you explain quantum computing?<|STK_SP|>

[Assistant]
"""
    tokens_src = tokenizer(context, return_tensors="pt").input_ids
    tokens_src = torch.cat([torch.full((1,args.soft_emb_tokens), -100), tokens_src], 1) if args.soft_emb_tune else tokens_src
    tokens_src = tokens_src.to(model.device)
    # for n, p in model.named_parameters():
    #     print(n, p.device)
    # TODO(mrrsteyk): this looks dumb
    MIN_LEN = 100
    EOS = 0
    END = 50277
    TRIALS = 1 if args.greedy else 3
    for _ in range(TRIALS):
        print("--- --- ---")
        tokens = tokens_src.clone()
        for i in range(767):
            logits = model(tokens).float()
            logits = logits.view(-1, logits.size(-1))
            logits = logits[-1] # ???
            # print(logits, logits.shape)
            
            if not args.greedy:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                # print(probs, probs.shape)
                
                sorted_probs = torch.sort(probs, descending=True)[0]
                # print("sorted", sorted_probs, sorted_probs.shape)

                cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
                cutoff = float(sorted_probs[np.argmax(cumulative_probs > 0.99)])
                probs[probs < cutoff] = 0
                # print("cut probs", probs, probs.shape)
                if True:
                    probs[EOS] = 0
                if i < MIN_LEN:
                    probs[EOS] = 0
                    probs[END] = 0
                
                try:
                    out = torch.multinomial(probs.float(), num_samples=1)[0]
                except:
                    out = EOS
                # print(out, out.shape)
            else:
                out = torch.argmax(logits)
            
            if out == EOS:
                print("<|BAD|>", end='')
            if out == EOS or out == END:
                break
            tokens = torch.cat([tokens, torch.full((1, 1), out, device=tokens.device, dtype=tokens.dtype)], 1)
            print(tokenizer.decode(out), end='', flush=True)
            # break
        print()
