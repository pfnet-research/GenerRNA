The newest version with trained model is placed in the following repository:
https://huggingface.co/pfnet/GenerRNA

# GenerRNA
GenerRNA is a generative RNA language model based on a Transformer decoder-only architecture. It was pre-trained on 30M sequences, encompassing 17B nucleotides.

Here, you can find all the relevant scripts for running GenerRNA on your machine. GenerRNA enable you to generate RNA sequences in a zero-shot manner for exploring the RNA space, or to fine-tune the model using a specific dataset for generating RNAs belonging to a particular family or possessing specific characteristics.

# Requirements
A CUDA environment, and a minimum VRAM of 8GB was required.
### Dependencies
```
torch>=2.0
numpy
transformers==4.33.0.dev0
datasets==2.14.4
tqdm
```

# Usage
Firstly, combine the split model using the command `cat model.pt.part-* > model.pt.recombined`
#### Directory tree
```
.
├── LICENSE
├── README.md
├── configs 
│   ├── example_finetuning.py
│   └── example_pretraining.py
├── experiments_data
├── model.pt.part-aa # splited bin data of pre-trained model
├── model.pt.part-ab
├── model.pt.part-ac
├── model.pt.part-ad
├── model.py         # define the architecture
├── sampling.py      # script to generate sequences
├── tokenization.py  # preparete data
├── tokenizer_bpe_1024
│   ├── tokenizer.json
│   ├── ....
├── train.py # script for training/fine-tuning
```

### De novo Generation in a zero-shot fashion
Usage example:
```
python sampling.py \
    --out_path {output_file_path} \
    --max_new_tokens 256 \
    --ckpt_path {model.pt} \
    --tokenizer_path {path_to_tokenizer_directory, e.g /tokenizer_bpe_1024}
```
### Pre-training or Fine-tuning on your own sequences
First, tokenize your sequence data, ensuring each sequence is on a separate line and there is no header.
```
python tokenization.py \
    --data_dir {path_to_the_directory_containing_sequence_data} \
    --file_name {file_name_of_sequence_data} \
    --tokenizer_path {path_to_tokenizer_directory}  \
    --out_dir {directory_to_save_tokenized_data} \
    --block_size 256
```

Next, refer to `./configs/example_**.py` to create a config file of GPT model.

Lastly, excute following command:
```
python train.py \
    --config {path_to_your_config_file}
```

### Train your own tokenizer
Usage example:
```
python train_BPE.py \
    --txt_file_path {path_to_training_file(txt,each sequence is on a separate line)} \
    --vocab_size 50256 \
    --new_tokenizer_path {directory_to_save_trained_tokenizer} \
                
```

# License
The source code is licensed MIT. See `LICENSE`
