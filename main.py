'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

def text_generator(state_dict):
    text=input("Enter the text : ") 
    quiet=True
    nsamples=1
    batch_size=-1
    length=-1
    temperature=0.7
    unconditional='store_true'
    top_k=40
    #args.text=input("Enter the text : ")
    #parser = argparse.ArgumentParser()
    '''parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()'''

    if quiet is False:
        print(args)

    if batch_size == -1:
        batch_size = 1
    assert nsamples % batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    if length == -1:
        length = config.n_ctx // 2
    elif length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    print(text)
    context_tokens = enc.encode(text)

    generated = 0
    for _ in range(nsamples // batch_size):
        out = sample_sequence(
            model=model, length=length,
            context=context_tokens  if not  unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if unconditional else None,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(batch_size):
            generated += 1
            text = enc.decode(out[i])
            if quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)

if __name__ == '__main__':
    if os.path.exists('gpt2-pytorch_model.bin'):
        state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
        text_generator(state_dict)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()
