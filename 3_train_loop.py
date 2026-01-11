import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
BATCH_SIZE = 1          
GRAD_ACCUM_STEPS = 4    
LEARNING_RATE = 1e-4
EPOCHS = 10
EMBEDDING_DIM = 768     # nomic-embed-text size
HIDDEN_DIM = 256        
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. CORE BDH LAYER ---
class BDHPlasticity(nn.Module):
    def __init__(self, d_model, rank=16):
        super().__init__()
        self.d = d_model
        self.k = rank
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.eta   = nn.Parameter(torch.tensor(2e-4))
        # High retention to prevent forgetting
        self.lam_p = nn.Parameter(torch.tensor(0.995)) 
        
        self.register_buffer("U", torch.zeros(d_model, rank))
        self.register_buffer("V", torch.zeros(d_model, rank))

    def reset_state(self):
        self.U.zero_()
        self.V.zero_()

    def forward(self, h, update=True, freeze_weights=False):
        B, T, D = h.shape
        h_norm = F.normalize(h, dim=-1)
        
        # READ: Compute Activation (Energy) based on current memory
        A_h = h_norm @ self.V @ self.U.T
        out = h + torch.tanh(self.alpha) * A_h.detach()
        
        # WRITE: Hebbian Update
        if update and not freeze_weights:
            with torch.no_grad():
                lam = torch.sigmoid(self.lam_p)
                for t in range(T):
                    h_t = h_norm[:, t, :].reshape(B, D)
                    y_t = A_h[:, t, :].reshape(B, D)
                    
                    self.U.mul_(lam)
                    self.V.mul_(lam)
                    
                    # Update rule
                    self.U.add_(self.eta * h_t.T @ y_t[:, :self.k])
                    self.V.add_(self.eta * y_t.T @ h_t[:, :self.k])
                    
                    # Clamp for stability
                    self.U.clamp_(-0.2, 0.2)
                    self.V.clamp_(-0.2, 0.2)
        
        # Return both the processed output and the raw activation (A_h)
        # A_h acts as a "Surprise" or "Resonance" signal
        return out, A_h 

# --- 2. PERSISTENT CRITIC MODEL ---
class PersistentJudge(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.compressor = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Path A: The Critic (Learns Backstory, then Freezes)
        self.critic = BDHPlasticity(d_model=hidden_dim, rank=32)
        
        # Path B: The Observer (Learns continuously)
        self.observer = BDHPlasticity(d_model=hidden_dim, rank=32)
        
        # Classifier Inputs: [Critic_Max, Observer_Max, Entity_Score]
        # Dims: 256 + 256 + 1 = 513
        self.classifier = nn.Linear(hidden_dim * 2 + 1, 1)

    def forward(self, backstory, novel, entity_score):
        # Reset Memories
        self.critic.reset_state()
        self.observer.reset_state()
        
        # Compress Inputs
        b_emb = self.dropout(torch.relu(self.compressor(backstory)))
        n_emb = self.dropout(torch.relu(self.compressor(novel)))
        
        # --- IMPRINT PHASE (Backstory) ---
        # Both layers learn the backstory first
        _, _ = self.critic(b_emb, update=True)
        _, _ = self.observer(b_emb, update=True)
        
        # --- JUDGMENT PHASE (Novel) ---
        # 1. Critic: Reads novel with FROZEN weights. 
        # If novel matches backstory logic -> High Resonance.
        # If novel contradicts -> Low Resonance / High Surprise.
        _, critic_activations = self.critic(n_emb, update=False, freeze_weights=True)
        
        # 2. Observer: Reads novel and continues learning (tracking plot changes)
        observer_out, _ = self.observer(n_emb, update=True, freeze_weights=False)
        
        # --- POOLING ---
        # Take the maximum signal across the whole book (did ANY part resonate/conflict?)
        crit_pool, _ = torch.max(critic_activations, dim=1) # [Batch, Hidden]
        obs_pool, _ = torch.max(observer_out, dim=1)        # [Batch, Hidden]
        
        # --- CLASSIFY ---
        # Combine Neural Signals with Symbolic Entity Score
        combined = torch.cat([crit_pool, obs_pool, entity_score.unsqueeze(1)], dim=1)
        
        return self.classifier(combined)

# --- 3. DATASET LOADER ---
class HackathonDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.sample_ids = []
        # Find all valid samples based on label files
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith("_label.npy"):
                    self.sample_ids.append(f.replace("_label.npy", ""))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        
        # Paths
        b_path = os.path.join(self.data_dir, f"{sid}_backstory.npy")
        n_path = os.path.join(self.data_dir, f"{sid}_novel.npy")
        l_path = os.path.join(self.data_dir, f"{sid}_label.npy")
        e_path = os.path.join(self.data_dir, f"{sid}_entity_score.npy") # New file
        
        # Load
        backstory = torch.tensor(np.load(b_path)).float()
        novel = torch.tensor(np.load(n_path)).float()
        label = torch.tensor(np.load(l_path)).float()
        
        # Load Entity Score (fallback to 0.5 if missing)
        if os.path.exists(e_path):
            entity_score = torch.tensor(np.load(e_path)).float()
        else:
            entity_score = torch.tensor([0.5]).float()
        
        # Fix Dimensions [Time, Dim] -> Add batch dim in DataLoader later
        if len(backstory.shape) == 1: backstory = backstory.unsqueeze(0)
        if len(novel.shape) == 1: novel = novel.unsqueeze(0)
            
        return backstory, novel, label, entity_score.squeeze()

# --- 4. TRAINING LOOP ---
def train():
    data_dir = "data_embeddings"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("Error: data_embeddings folder is empty. Run Step 0 first!")
        return

    dataset = HackathonDataset(data_dir)
    print(f"Found {len(dataset)} training samples.")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Persistent Judge
    model = PersistentJudge(input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    print("Starting training (Persistent Critic Architecture)...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        
        for i, (b_story, novel, label, e_score) in enumerate(dataloader):
            # Move to GPU
            b_story = b_story.to(DEVICE)
            novel = novel.to(DEVICE)
            label = label.to(DEVICE)
            e_score = e_score.to(DEVICE)
            
            # Forward
            logits = model(b_story, novel, e_score)
            loss = criterion(logits.squeeze(), label.squeeze())
            
            # Backward
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            
            if i % 10 == 0:
                print(f"  Epoch {epoch+1} | Step {i} | Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f} ===")
        torch.save(model.state_dict(), f"bdh_judge_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()