# -----------------------------------------------------------------------------
# I/O

# learning data directory, train.bin and valid .bin are expected. You should prepare them using tokenize.py
data_dir = 'directory_containing_train.bin/val.bin'
out_dir = 'output_directory'  # output directory
log_dir = os.path.join(out_dir, 'logs') # logs will be written in to out_dir/logs

# -----------------------------------------------------------------------------
# model parameters
meta_vocab_size = 1024
block_size = 256
n_layer=24
n_head=16 
n_embd=1024 # 350M, medium
bias = False # do we use bias inside LayerNorm and Linear layers?

# -----------------------------------------------------------------------------
# learning parameters
max_iters = 1000000 # total number of training iterations
eval_interval = 100000
log_interval = 1
eval_iters = 100
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
gradient_accumulation_steps = 16 # used to simulate larger batch sizes, should be multiple of GPU number
batch_size = 16 

# adamw optimizer
learning_rate = 1e-3 # max learning rate
dropout = 0.0 
weight_decay = 0
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 5000 # how many steps to warm up for
lr_decay_iters = 1000000 # should be ~= max_iters per Chinchilla
min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float32' # if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
