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
#### Directory tree
```

```

Firstly, combine the split model using the command `cat model.pt.part-* > model.pt.recombined`

### De novo Generation in a zero-shot fashion

### Fine-tuning on your own sequences
#### Data preparation

#### Fine-tuning the model

