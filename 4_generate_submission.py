import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# We use Epoch 9 because it had the lowest average loss (0.6354)
MODEL_PATH = "bdh_judge_epoch_9.pt" 
DATA_DIR = "data_embeddings"
OUTPUT_FILE = "results.csv"

# Model Parameters (Must match training!)
EMBEDDING_DIM = 768
HIDDEN_DIM = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- REDEFINE MODEL (Must be identical to training script) ---
class BDHPlasticity(nn.Module):
    def __init__(self, d_model, rank=16):
        super().__init__()
        self.d = d_model
        self.k = rank
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.eta   = nn.Parameter(torch.tensor(2e-4))
        self.lam_p = nn.Parameter(torch.tensor(0.95))
        self.register_buffer("U", torch.zeros(d_model, rank))
        self.register_buffer("V", torch.zeros(d_model, rank))

    def reset_state(self):
        self.U.zero_()
        self.V.zero_()

    def forward(self, h, update=True):
        B, T, D = h.shape
        h_norm = F.normalize(h, dim=-1)
        A_h = h_norm @ self.V @ self.U.T
        out = h + torch.tanh(self.alpha) * A_h.detach()
        
        if update:
            with torch.no_grad():
                lam = torch.sigmoid(self.lam_p)
                for t in range(T):
                    h_t = h_norm[:, t, :].reshape(B, D)
                    y_t = A_h[:, t, :].reshape(B, D)
                    self.U.mul_(lam)
                    self.V.mul_(lam)
                    self.U.add_(self.eta * h_t.T @ y_t[:, :self.k])
                    self.V.add_(self.eta * y_t.T @ h_t[:, :self.k])
                    self.U.clamp_(-0.2, 0.2)
                    self.V.clamp_(-0.2, 0.2)
        return out

class ContextJudge(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.compressor = nn.Linear(input_dim, hidden_dim)
        # No dropout during inference
        self.memory = BDHPlasticity(d_model=hidden_dim, rank=32)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, backstory, novel):
        self.memory.reset_state()
        b_emb = torch.relu(self.compressor(backstory))
        self.memory(b_emb, update=True)
        n_emb = torch.relu(self.compressor(novel))
        final_states = self.memory(n_emb, update=True)
        pooled, _ = torch.max(final_states, dim=1) 
        return self.classifier(pooled)

# --- INFERENCE ---
def generate():
    print(f"Loading model from {MODEL_PATH}...")
    model = ContextJudge(input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Set to evaluation mode

    # Load Test Data IDs from the original CSV to ensure order
    test_df = pd.read_csv("test.csv")
    ids = test_df['id'].tolist()
    
    results = []
    
    print("Running prediction on test set...")
    with torch.no_grad():
        for story_id in tqdm(ids):
            # Construct file paths
            # Note: 0_prepare_data.py saved them as {id}_backstory.npy
            b_path = os.path.join(DATA_DIR, f"{story_id}_backstory.npy")
            n_path = os.path.join(DATA_DIR, f"{story_id}_novel.npy")
            
            if not os.path.exists(b_path) or not os.path.exists(n_path):
                print(f"Warning: Missing files for ID {story_id}. Defaulting to 0.")
                results.append({"Story ID": story_id, "Prediction": 0, "Rationale": "File missing"})
                continue
                
            # Load
            b_data = torch.tensor(np.load(b_path)).float().unsqueeze(0).to(DEVICE)
            n_data = torch.tensor(np.load(n_path)).float().unsqueeze(0).to(DEVICE)
            
            # Predict
            logits = model(b_data, n_data)
            prob = torch.sigmoid(logits).item()
            
            # Threshold at 0.5
            pred = 1 if prob > 0.5 else 0
            
            # Optional Rationale (Generic for Track B)
            rationale = "BDH Plasticity state consistent" if pred == 1 else "Plasticity convergence failure detected"
            
            results.append({
                "Story ID": story_id,
                "Prediction": pred,
                "Rationale": rationale
            })

    # Save
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Submission saved to {OUTPUT_FILE}")
    print(res_df.head())

if __name__ == "__main__":
    generate()