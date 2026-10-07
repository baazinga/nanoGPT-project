import torch
from model_with_kvcache import BigramLanguageModel
from data import load_dataset
from config import Config

# Evaluate minimal script
dataset = load_dataset()
vocab_size = dataset["vocab_size"]
decode = dataset["decode"]

model = BigramLanguageModel(vocab_size).to(Config.device)
m = model

# Load best model weights
state = torch.load("checkpoint_best_1.pt", map_location=str(Config.device))
model.load_state_dict(state)
print("Loaded best checkpoint.")

# Inference
context = torch.zeros((1, 1), dtype=torch.long).to(Config.device)

with torch.no_grad():
    out = model.generate(context, max_new_tokens=Config.max_new_tokens)

print(decode(out[0].tolist()))