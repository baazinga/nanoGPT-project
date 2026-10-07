import torch
import os
from config import Config

def save_checkpoint(model, optimizer, step, loss, path="checkpoint_last.pt"):
    state = {
        "step":step,
        "model_state": model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "loss":loss,
    }
    torch.save(state, path)
    print(f"[Checkpoint] Saved checkpoint to {path}")

def save_best_model(model, val_loss, best_loss, path="checkpoint_best.pt"):
    if val_loss < best_loss:
        torch.save(model.state_dict(), path)
        print(f"[Checkpoint] New best model saved to {path} (val_loss {val_loss:.4f})")
        return val_loss
    return best_loss

def load_checkpoint(path, model, optimizer=None, map_location=str(Config.device)):
    if not os.path.exists(path):
        print(f"[Checkpoint] no checkpoint at {path}")
        return None
    state = torch.load(path,map_location=map_location, weights_only=True)
    # Support both the current full checkpoint and earlier weights-only files.
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    print(f"[Checkpoint] Loaded checkpoint from {path} (step {state.get('step', 'N/A')})")
    return state
