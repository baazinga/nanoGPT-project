import torch
import torch.nn as nn
from torch.nn import functional as F


#parameters
batch_size = 12 #number of sequences that will be processing in parallel
block_size = 64 #the maximum context length of predictions
max_iters = 5000 #training loop
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
max_new_tokens = 500
#------------------

torch.manual_seed(1337)


with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()

print ("length of dataset in characters:",len(text))

#all the characters that would occur in the text
chars = sorted (list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#creating a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch  in enumerate (chars)}
encode = lambda s:[stoi[c] for c in s] #encode:take a string, out out as a list of numbers
decode = lambda l:''.join([itos[i] for i in l]) #decoder: take a list of numbers, out put as a string

#encode the entire text datasetand store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print (data.shape, data.dtype)

#seperate the data set into train and validation
n = int(0.9*len(data)) #90%train rest val
train_data = data[:n]
val_data = data[n:]

train_data[:block_size+1] #time dimension

x = train_data[:block_size] #the first block sized input characters
y = train_data[1:block_size+1] #the next~
for t in range(block_size):
    context = x[:t+1]
    target = y[t]

def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch .stack([data[i+1:i+block_size+1] for i in ix]) #become a row of 4x8 tensor
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q@k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)

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
            nn.Dropout(dropout),
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

    def forward(self, x):
        x = x + self.sa(self.n1(x)) #Residual Network
        x = x + self.ffwd(self.n2(x))
        return x


#super simple BigramModel
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token frm a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #position embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layernorm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B,T = idx.shape

        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) #(B,T,C) batch time channel
        pos_emb = self.position_embedding_table(torch.arange(T)) #(T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) #(B,T,C') C'!=C  C'=vocab_size

        if targets is None:
            loss = None 
        else:
             B, T, C = logits.shape
             logits = logits.view(B*T,C)
             targets = targets.view(B*T)
             loss = F.cross_entropy(logits, targets) 

        return logits,loss
    
    #generation of the model: take a (b,t) generate (b,t+i)
    def generate(self, idx, max_new_tokens) :
    #idx is(B,T)array of indices in the current batch
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
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


model = BigramLanguageModel()
m = model

#the model is fixed so we are going to train the model
optimizer = torch.optim.Adam(m.parameters(),lr=learning_rate)

for iter in range (max_iters):

    #every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    if iter == max_iters-1 :
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    #sample a batch
    xb,yb = get_batch('train')

   #evaluate the loss
    logits,loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=max_new_tokens)[0].tolist())) 