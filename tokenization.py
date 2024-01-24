import os
import numpy as np
from transformers import AutoTokenizer
import random
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process the text data for tokenization.')
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of the raw data.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the trained AutoTokenizer.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory of output files.")
    parser.add_argument("--end_with_eos", type=bool, default=True, help="Whether each line ends with `eos_token`.")
    return parser.parse_args()

def shuffle_and_split_data(file_path, split_ratio=0.95):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)
    split_at = int(split_ratio * len(lines))
    return lines[:split_at], lines[split_at:]

def write_to_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line.replace(' ', ''))

def tokenize_lines(tokenizer, lines, end_with_eos, block_size = 1e10):
    tokenized_ids = []
    for i, line in enumerate(lines):
        if not end_with_eos:
            line = line.strip() + tokenizer.eos_token
        ids = tokenizer.encode(line)

        #block size limitation
        if len(ids) <= block_size-1:
            tokenized_ids.extend(ids) 
        
        if (i + 1) % 100000 == 0: 
            print(f"Processed {i + 1} lines.")
    return tokenized_ids

def save_tokenized_data(tokenized_data, file_path):
    np_data = np.array(tokenized_data, dtype=np.uint16)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np_data.tofile(file_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of raw data & output files")
    parser.add_argument("--file_name", type=str, default="data.txt",required=True)
    parser.add_argument("--out_dir", type=str, required=False, help="directory of output files(default=data_dir). A train.bin and a valid.bin will be built and expect to be used in train.py")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to a trained AutoTokenizer")
    parser.add_argument("--block_size", type=str, required=True, help="Max token length")
    args = parser.parse_args()

    # Paths setup
    if args.out_dir is None:
        out_dir = args.data_dir
    else:
        out_dir = args.out_dir
    raw_data_path = os.path.join(args.data_dir, args.file_name)
    train_txt_path = os.path.join(out_dir, 'train.txt')
    val_txt_path = os.path.join(out_dir, 'val.txt')
    train_bin_path = os.path.join(out_dir, 'train.bin')
    val_bin_path = os.path.join(out_dir, 'val.bin')
    print("Paths setup complete...")

    # Data preparation
    train_lines, val_lines = shuffle_and_split_data(raw_data_path)
    write_to_file(train_txt_path, train_lines)
    write_to_file(val_txt_path, val_lines)
    print("Data preparation complete...")

    # Tokenization
    end_with_eos = False
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    train_ids = tokenize_lines(tokenizer, train_lines, end_with_eos, int(args.block_size))
    val_ids = tokenize_lines(tokenizer, val_lines, end_with_eos, int(args.block_size))
    print("Tokenization complete...")

    # Save tokenized data
    save_tokenized_data(train_ids, train_bin_path)
    save_tokenized_data(val_ids, val_bin_path)
    print("Tokenized data saved...")

if __name__ == "__main__":
    main()
