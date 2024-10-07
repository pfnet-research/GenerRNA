"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from tqdm import tqdm
import random
import numpy as np
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
import argparse
import itertools
import random


parser = argparse.ArgumentParser()
parser.add_argument("--init_from", type=str, default="resume", help="Directory of raw data & output files")
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--num_samples", type=int, required=False, default=100000)
parser.add_argument("--max_new_tokens", type=int, required=True, help="number of tokens generated in each sample")
parser.add_argument("--strategy",type=str, required=False,default='top_k',help="should be in ['greedy_search', 'sampling', 'top_k', 'beam_search']")
parser.add_argument("--beam_size",type=int, required=False,default=3,help="beam size for beam search")
parser.add_argument("--temperature",type=float, required=False,default=1.0,help="1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions")
parser.add_argument("--top_k",type=int, required=False,default=20,help="retain only the top_k most likely tokens, clamp others to have 0 probability")
parser.add_argument("--ckpt_path",type=str, required=True,help="path to a checkpoint/model")
parser.add_argument("--tokenizer_path",type=str, required=True,help="path to a tokenizer directory")
parser.add_argument("--start",type=str, required=False,default="<|endoftext|>")
parser.add_argument("--repetition_penalty",type=float, required=False,default=1.0)
parser.add_argument("--shuffle_token", action='store_true', help="Enable shuffling of tokens before decoding")
parser.add_argument("--fasta", action='store_true', default=True, help="Enable writing output in FASTA format")

args = parser.parse_args()
init_from = args.init_from
out_path = args.out_path
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens
strategy = args.strategy
assert strategy in ['greedy_search', 'sampling', 'top_k', 'beam_search']
beam_size = args.beam_size
temperature = args.temperature
top_k = args.top_k
ckpt_path = args.ckpt_path
tokenizer_path = args.tokenizer_path
start = args.start
repetition_penalty = args.repetition_penalty
fasta = args.fasta


# -----------------------------------------------------------------------------
seed = random.randint(1,6666)
device = 'cuda' 
dtype = 'float32'
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
# Load tokenizer & ckpt_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
encode = tokenizer.encode
decode = tokenizer.decode

fasta_out_path = os.path.splitext(out_path)[0] + ".fasta" if fasta else None

if strategy in["sampling", "top_k"]:
    start_ids = encode("".join(start))
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])


    with open(out_path, 'a') as f:
        with open(fasta_out_path, 'a') if fasta else nullcontext() as fasta_f:
            with torch.no_grad():
                with ctx:
                    for k in tqdm(range(num_samples), desc="Generating samples"):
                        token_sequence = model.generate(x, max_new_tokens, strategy=strategy, temperature=temperature, top_k=top_k, repetition_penalty=repetition_penalty)[0].tolist()
                        
                        # Shuffle tokens if --shuffle_token is specified
                        if args.shuffle_token:
                            random.shuffle(token_sequence)

                        y = decode(token_sequence).replace(' ', '')
                        # y = decode(token_sequence).replace('\n', '').replace(' ', '') + '\n'
                        f.write(y)
                        f.flush()


                        if fasta:
                            fasta_entry = f">sample_{k}\n{y.replace(' ', '')}\n"
                            fasta_f.write(fasta_entry.strip() + '\n')
                            fasta_f.flush()


elif strategy in ["beam_search", "greedy_search"]:
    with open(out_path, 'a') as f:
        with open(fasta_out_path, 'a') if fasta else nullcontext() as fasta_f:
            with torch.no_grad():
                with ctx:
                    start = '<|endoftext|>'
                    start_ids = encode(start)
                    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

                    token_sequence = model.generate(x, max_new_tokens, strategy=strategy, temperature=temperature, top_k=top_k, repetition_penalty=repetition_penalty, beam_size=beam_size)[0].tolist()

                    y = decode(token_sequence).replace(' ', '')
                    f.write(y)
                    f.flush()


                    if fasta:
                        fasta_entry = f">sample_{k}\n{y.replace(' ', '')}\n"
                        fasta_f.write(fasta_entry.strip() + '\n')
                        fasta_f.flush()