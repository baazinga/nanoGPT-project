#config
import torch
class Config:
    #data
    block_size = 2048   #the maximum context length of predictions
    train_split = 0.9
    vocab_size = None

    #training
    batch_size = 12   #number of sequences that will be processing in parallel
    max_iters = 5000   #training loop
    eval_interval = 500
    learning_rate = 1e-3
    eval_iters = 200

    #model
    n_embd = 256
    n_head = 4
    n_layer = 4
    dropout = 0.2
    max_new_tokens = 500

    # device setup (choose GPU if available, else CPU)
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device)

    #KVcache flag
    #use_kv_cache = False  # default: OFF for training
    use_kv_cache = True
