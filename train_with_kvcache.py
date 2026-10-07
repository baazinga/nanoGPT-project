import torch
import torch.nn as nn
from torch.nn import functional as F

from model_with_kvcache import BigramLanguageModel
from utils import get_batch, estimate_loss
from data import load_dataset
from config import Config
from checkpoint import save_checkpoint, save_best_model, load_checkpoint


#device
device = Config.device
print("Using device:", device)

# load dataset
dataset = load_dataset()
train_data = dataset["train"]
val_data = dataset["val"]
vocab_size = dataset["vocab_size"]
encode = dataset['encode']
decode = dataset['decode']

#init model
model = BigramLanguageModel(
    vocab_size
).to(Config.device)
m = model
#the model is fixed so we are going to train the model
optimizer = torch.optim.AdamW(m.parameters(),lr=Config.learning_rate)

# checkpoint files
CKPT_LAST = "checkpoint_last_1.pt"
CKPT_BEST = "checkpoint_best_1.pt"


start_step = 0
best_val_loss = float("inf")

state = load_checkpoint(CKPT_LAST, model, optimizer, map_location=Config.device)
if state is not None:
    start_step = int(state.get("step", 0))
    if "loss" in state and state["loss"] is not None:
        print(f"[Resume] loaded last train loss = {state['loss']:.4f} (resuming at step {start_step})")

#training loop
for iter in range (start_step, Config.max_iters):

    #every once in a while evaluate the loss on train and val sets
    if iter % Config.eval_interval == 0:
        losses = estimate_loss(model,train_data, val_data)
        train_loss = float(losses["train"])
        val_loss = float(losses["val"])

        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Save best model weights (weights-only) if val improves
        best_val_loss = save_best_model(model, val_loss, best_val_loss, path=CKPT_BEST)
        # Save full training checkpoint (for resuming)
        save_checkpoint(model, optimizer, iter, train_loss, path=CKPT_LAST)

    #the last step in training
    if iter == Config.max_iters-1 :
        losses = estimate_loss(model,train_data, val_data)
        train_loss = float(losses["train"])
        val_loss = float(losses["val"])
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    #sample a batch
    xb,yb = get_batch('train', train_data, val_data)

    #evaluate the loss
    logits, loss = m(xb, use_cache=False, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# final save
save_checkpoint(model, optimizer, iter, float(loss), path=CKPT_LAST)
save_best_model(model, val_loss, best_val_loss, path=CKPT_BEST)