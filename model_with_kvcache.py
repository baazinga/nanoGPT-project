
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import Config

#TO DO                                          CHECK
#1. add kv cahe in single head
#       1.1 where to add flag = use_cache?      done
#       1.2 when to reset?
#       1.3 mask miss size
#2. modify params passing in multi-head
#       2.1 append using kvcahe (flag: use_cache)
#.      2.2 缓存拼接操作应在 no_grad 下执行 & 需要截断（防止无限增长 / OOM）
#3. modify BigramModel for kvcach implement
#       3.1 cannot use Sequential if using KV cache?
#4. modify generate scheme
#5. Position embeding
#   absolute position embeding causing the model to confuse with token index(sequence number),
#   therefore generating results with kvcache are worse than without.



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

        # Causal mask
        self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(Config.dropout)

        ####### NEW ######### add kvcahe buffer
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        ####### NEW #########

    def forward(self, x, use_cache=False):
        B,T,C = x.shape

        #compute k,q,v -> k_new, v_new
        #k = self.key(x)
        #v = self.value(x)
        k_new = self.key(x)
        v_new = self.value(x)
        q = self.query(x)

        ######################## NEW ################################
        if use_cache: #use_cache is a flag
            # update cache without tracking gradients
            with torch.no_grad():
                # First generation step: no cache yet
                if self.cache_k is None:
                    # ensure cache is on same device
                    self.cache_k = k_new.detach().to(x.device)
                    self.cache_v = v_new.detach().to(x.device)
                else:
                    self.cache_k = torch.cat([self.cache_k, k_new.detach().to(self.cache_k.device)], dim=1)
                    self.cache_v = torch.cat([self.cache_v, v_new.detach().to(self.cache_v.device)], dim=1)

                # keep only the most recent Config.block_size tokens to avoid unbounded growth
                if self.cache_k.size(1) > Config.block_size:
                    self.cache_k = self.cache_k[:, -Config.block_size:, :].contiguous()
                    self.cache_v = self.cache_v[:, -Config.block_size:, :].contiguous()

            k, v = self.cache_k, self.cache_v
        else:
            k, v = k_new, v_new
        ######################## NEW ################################


        # scale by head dim (not full embedding)
        head_dim = k.size(-1)
        scale = head_dim ** -0.5

        # attention weights: q @ k^T  -> shape (B, T_q, T_k)
        wei = q @ k.transpose(-2, -1) * scale

        # Build a causal mask of shape (T_q, T_k). With a cache, the query
        # positions start after the cached prefix rather than at position zero.
        T_q = q.size(1)
        T_k = k.size(1)
        if use_cache:
            past_len = T_k - T_q
            query_positions = torch.arange(
                past_len, past_len + T_q, device=wei.device
            ).unsqueeze(1)
            key_positions = torch.arange(T_k, device=wei.device).unsqueeze(0)
            mask = key_positions <= query_positions
        else:
            mask = self.tril[:T_q, :T_k].to(wei.device).bool()
        wei = wei.masked_fill(mask == 0, float("-inf"))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out

    ######################## NEW ################################
    def reset_cache(self): #when to reset?
        self.cache_k = None
        self.cache_v = None
    ######################## NEW ################################

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        n_embd = Config.n_embd

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x, use_cache=False):
        ######################## NEW modified ################################
        out = torch.cat([h(x, use_cache=use_cache) for h in self.heads], dim=-1)
        ######################## NEW modified################################
        out = self.dropout(self.proj(out))
        return out

    ######################## NEW ################################
    def reset_cache(self):
        for head in self.heads:
            head.reset_cache()
    ######################## NEW ################################

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

    def forward(self, x, use_cache=False):
        ######################## NEW modified ################################
        sa_out = self.sa(self.n1(x),use_cache=use_cache)
        ######################## NEW modified ################################
        x = x + sa_out #Residual Network
        x = x + self.ffwd(self.n2(x))
        return x

    ######################## NEW ################################
    def reset_cache(self):
        self.sa.reset_cache()
    ######################## NEW ################################


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

        ############## NEW ################
        # cannot use Sequential if using KV cache?
        #self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) # final layernorm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, use_cache=Config.use_kv_cache,targets=None):
        B,T = idx.shape

        ######################## NEW ################################
        # compute past length from cache if using cache
        past_len = 0
        if use_cache:
            # try to read cache length from first head of first block (if exists)
            try:
                first_head_cache = self.blocks[0].sa.heads[0].cache_k
                if first_head_cache is not None:
                    # cache_k shape: (B, past_len, head_size)
                    past_len = first_head_cache.size(1)
            except Exception:
                past_len = 0

        # absolute position ids (account for past length)
        pos_ids = torch.arange(past_len, past_len + T, device=idx.device, dtype=torch.long)  # (T,)

        # 检查 pos id 是否会越界
        if (past_len + T) > Config.block_size:
            raise RuntimeError(f"Context (past_len + T = {past_len + T}) exceeds block_size ({Config.block_size}). "
                            "Increase Config.block_size or shorten prompt/generation length.")

        # expand to batch dim so shapes align
        pos_emb = self.position_embedding_table(pos_ids).unsqueeze(0).expand(B, -1, -1)  # (B,T,C)

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        x = tok_emb + pos_emb

        # run blocks (each block expects use_cache flag)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        ######################## NEW ################################

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits,loss


    ######################## NEW ################################
    def reset_cache(self):
        for block in self.blocks:
                block.reset_cache()
    ######################## NEW ################################

    ######################## NEW ################################
    def generate(self, idx, max_new_tokens) :
        use_cache = Config.use_kv_cache

        if use_cache:
            self.reset_cache()
            with torch.no_grad():
                # Prime once with the full prompt. Its final logit predicts the
                # first new token, so the last prompt token is not duplicated.
                logits, _ = self(idx, use_cache=True)

        for step in range(max_new_tokens):
            if not use_cache:
                idx_cond = idx[:, -Config.block_size:]
                with torch.no_grad():
                    logits, _ = self(idx_cond, use_cache=False)

            probs = F.softmax(logits[:, -1, :],dim=-1)#->(B,V)
            idx_next = torch.multinomial(probs, num_samples=1)#->(B,1)
            idx = torch.cat ((idx, idx_next),dim=1)#->(B,T+1)

            if use_cache and step < max_new_tokens - 1:
                with torch.no_grad():
                    logits, _ = self(idx_next, use_cache=True)

        return idx
    ####################### NEW ################################


#    #generation of the model: take a (b,t) generate (b,t+i)
#    def generate(self, idx, max_new_tokens) :
#    #idx is(B,T)array of indices in the current batch
#        for _ in range(max_new_tokens):
#            idx_cond = idx[:, -Config.block_size:]
#            #get the predictions
#            logits, loss = self(idx_cond)
#            #focus only on the last time step
#            logits = logits[:,-1,:] #->(B,C)
#            #softmax
#            probs = F.softmax(logits,dim=-1)#->(B,C)
#            idx_next = torch.multinomial(probs, num_samples=1)#->(B,1)
#            #append sample index to the running sequence
#            idx = torch.cat ((idx, idx_next),dim=1)#->(B,T+1)
#        return idx
