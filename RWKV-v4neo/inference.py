# !!! MAKESHIFT INFERENCE SCRIPT !!!

import argparse
import os
import torch
import numpy as np

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint, get_fp32_state_dict_from_zero_checkpoint

from soft_embedding_hotswap import SoftEmbedding

from transformers import PreTrainedTokenizerFast

def get_args():
    parser = argparse.ArgumentParser()

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

if __name__ == "__main__":
    args = get_args()
    print(args)

    assert args.precision in ["32", "16", "bf16"]
    if args.precision == "16":
        args.precision = 16
        os.environ["RWKV_FLOAT_MODE"] = "fp16"
    else:
        os.environ["RWKV_FLOAT_MODE"] = str(args.precision)
    os.environ["RWKV_T_MAX"] = str(args.ctx_len + args.soft_emb_tokens if args.soft_emb_tune else 0)

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
        args.layerwise_lr = False
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
            print(state_dict.keys())
            #model.emb.learned_embedding = state_dict["_forward_module.emb.learned_embedding"]
            model.load_state_dict(state_dict, strict=False)
    
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
    print("--- --- ---")
    model.freeze()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='../20B_tokenizer_openchatgpt.json')
    context = """quality: high

[System]
Assistant is a distilled language model trained by the community.<|STK_SP|>

[System]
<|STK_SP|>

[User]
Can you explain quantum computing<|STK_SP|>

[Assistant]
"""
    tokens = tokenizer(context, return_tensors="pt").input_ids
    tokens = torch.cat([torch.full((1,args.soft_emb_tokens), -100), tokens], 1)
    tokens = tokens.to(model.device)
    # for n, p in model.named_parameters():
    #     print(n, p.device)
    # TODO(mrrsteyk): this looks dumb
    MIN_LEN = 100
    EOS = 0
    END = 50277
    for i in range(767):
        logits = model(tokens).float()
        logits = logits.view(-1, logits.size(-1))
        logits = logits[-1] # ???
        # print(logits, logits.shape)
        
        if True:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # print(probs, probs.shape)
            
            sorted_probs = torch.sort(probs, descending=True)[0]
            # print("sorted", sorted_probs, sorted_probs.shape)

            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > 0.95)])
            probs[probs < cutoff] = 0
            # print("cut probs", probs, probs.shape)
            if i < MIN_LEN:
                probs[EOS] = 0
                probs[END] = 0
            
            out = torch.multinomial(probs.float(), num_samples=1)[0]
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