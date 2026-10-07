# file: model.py
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import Config

#Single attentionhead
class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        n_embd = Config.n_embd
        block_size = Config.block_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x):
        B,T,C = x.shape

        #compute k,q,v
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)

        # scale by head dim (not full embedding)
        head_dim = k.size(-1)
        scale = head_dim ** -0.5

        wei = q @ k.transpose(-2, -1) * scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        n_embd = Config.n_embd

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd),
            nn.Dropout(Config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) #computation is done by multi-head attention
        self.ffwd = FeedForward(n_embd) #communication is done by a feedforward network
        self.n1 = nn.LayerNorm(n_embd)
        self.n2 = nn.LayerNorm(n_embd)

    def forward(self, x, past_kv=None, use_cache=False):
        sa_out = self.sa(self.n1(x))
        x = x + sa_out #Residual Network
        x = x + self.ffwd(self.n2(x))
        return x


#super simple BigramModel
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        n_embd = Config.n_embd
        block_size = Config.block_size
        n_layer = Config.n_layer
        n_head = Config.n_head

        #each token directly reads off the logits for the next token frm a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #position embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layernorm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        B,T = idx.shape

        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) #(B,T,C) batch time channel
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) #(T,C)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits,loss

    #generation of the model: take a (b,t) generate (b,t+i)
    def generate(self, idx, max_new_tokens) :
    #idx is(B,T)array of indices in the current batch
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -Config.block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:,-1,:] #->(B,C)
            #softmax
            probs = F.softmax(logits,dim=-1)#->(B,C)
            idx_next = torch.multinomial(probs, num_samples=1)#->(B,1)
            #append sample index to the running sequence
            idx = torch.cat ((idx, idx_next),dim=1)#->(B,T+1)
        return idx
