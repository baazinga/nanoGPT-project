import torch
from config import Config


def get_batch(split, train_data, val_data):
    block_size = Config.block_size
    batch_size = Config.batch_size
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size -1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch .stack([data[i+1:i+block_size+1] for i in ix]) #become a row of 4x8 tensor
    x,y = x.to(Config.device), y.to(Config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model,train_data, val_data):
    out = {}
    model.eval()

    for split in ['train','val']:
        losses = torch.zeros(Config.eval_iters)
        for k in range(Config.eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            #cache forbidden!
            logits, loss= model(X, use_cache=False, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out