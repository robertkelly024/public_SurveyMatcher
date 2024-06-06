import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple
import sqlite3
import pandas as pd
import openpyxl
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import models
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten, tree_map



def build_parser():
    parser = argparse.ArgumentParser(description="Model Inference with Adapters.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=150,
        help="The maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    parser.add_argument(
        "--xlfile",
        "-p",
        type=str,
        help="The path to the local excel file of JDs",
        default="realJDs_sample.xlsx",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Path for the trained adapter weights.",
    )
    return parser

def generate(model, prompt, tokenizer, args):
    print(prompt, end="", flush=True)

    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    for token, n in zip(
        models.generate(prompt, model, args.temp),
        range(args.max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        print(s[skip:], end="", flush=True)
        skip = len(s)

    print(tokenizer.decode(tokens)[skip:], flush=True)
    print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
    return s

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(0)

    print("Loading pretrained model")
    model, tokenizer, _ = models.load(args.model)

    # Freeze all layers other than LORA linears
    model.freeze()
    for l in model.model.layers[len(model.model.layers) - args.lora_layers :]:
        l.self_attn.q_proj = models.LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = models.LoRALinear.from_linear(l.self_attn.v_proj)
    
    
    # Load the adapter weights
    if not Path(args.adapter_file).is_file():
        raise ValueError(
            f"Adapter file {args.adapter_file} missing. "
        )
    model.load_weights(args.adapter_file, strict=False)

# start looping through jd db
    if args.xlfile is not None:
        db_path = args.xlfile
        full_db = pd.readexcel(db_path)

        columns = ["responsibilities", "skills", "experience"]
        for column in columns:
            full_db[column] = ''
        #loop the rows to send to fine-tuned model
        for index, row in full_db.iterrows():
            for column in columns:
                JD = row['description_cleaned']
                prompt = f"<s>[INST]What is or are the {column} listed in the following job description?\n{JD}\n[/INST]"
                description_converted = generate(model, prompt, tokenizer, args)
                full_db.at[index, column] = description_converted

        full_db.to_excel("output.xlsx", index=False)