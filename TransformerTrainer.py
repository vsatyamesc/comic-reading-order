"""
You might need to change orders.append(order) at Line 65, labelImg, I'm not exactly following Yolo Format for this, the 0th index stores the sequence, LabelImg stores class_id in 0th index, however LabelImg stores the data in the sequence you label them, so you can code it.
The parameters set here are the perfect ones I found after multiple training testing, even working on Real Mangas.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 5
MAX_PANELS = 20  # Fixed size for panels

hyperparams = {
    'BATCH_SIZE': 32,
    'NUM_EPOCHS': 50,
    'LEARNING_RATE': '1e-4',
    'D_MODEL': 128,
    'NHEAD': 4,
    'NUM_LAYERS': 5
}

def manga_collate_fn(batch):
    """Handle fixed-size panel sequences"""
    return {
        'coords': torch.stack([item['coords'] for item in batch]),
        'order': torch.stack([item['order'] for item in batch])
    }

class MangaDataset(Dataset):
    def __init__(self, data_dir, max_panels=MAX_PANELS):
        self.samples = []
        self.max_panels = max_panels  # Enforce fixed size
        
        for file in os.listdir(data_dir):
            if file.endswith('.txt'):
                coords = []
                orders = []
                with open(os.path.join(data_dir, file)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                order = int(parts[0])
                                xc, yc, w, h = map(float, parts[1:5])
                                x1 = xc - w/2
                                y1 = yc - h/2
                                x2 = xc + w/2
                                y2 = yc + h/2
                                coords.append([x1, y1, x2, y2, xc, yc, w, h])
                                orders.append(order)
                            except:
                                continue
                
                # Truncate and pad to fixed size
                coords = coords[:self.max_panels]  # Truncate if too long
                orders = orders[:self.max_panels]
                
                # Pad with zeros and -1
                pad_len = self.max_panels - len(coords)
                coords += [[0, 0, 0, 0, 0, 0, 0, 0]] * pad_len
                orders += [-1] * pad_len
                
                self.samples.append({
                    'coords': torch.FloatTensor(coords),
                    'order': torch.LongTensor(orders)
                })

    def __add__(self, other):
        """Combine two datasets"""
        combined = MangaDataset.__new__(MangaDataset)
        combined.max_panels = self.max_panels
        combined.samples = self.samples + other.samples
        return combined
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def create_augmented_copy(self, skip_prob=0.5):
        """
        Creates an augmented version of the dataset by randomly skipping panels
        Args:
            skip_prob: Probability of skipping each panel (0-1)
        Returns:
            New MangaDataset instance with augmented samples
        """
        augmented = MangaDataset.__new__(MangaDataset)
        augmented.max_panels = self.max_panels
        augmented.samples = []
        
        for sample in self.samples:
            # Create multiple variants per sample
            for _ in range(2):  # Generate 2 augmented versions per sample
                coords = sample['coords'].clone()
                orders = sample['order'].clone()
                
                # Randomly select panels to skip (1-6 panels)
                num_to_skip = random.randint(1, min(6, (orders != -1).sum().item()))
                valid_indices = [i for i,o in enumerate(orders) if o != -1]
                
                if len(valid_indices) > 1:  # Need at least 2 panels to skip
                    skip_indices = random.sample(valid_indices, num_to_skip)
                    
                    # Zero out skipped panels
                    for idx in skip_indices:
                        coords[idx] = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0])
                        orders[idx] = -1
                    
                    # Re-normalize orders to maintain sequence
                    remaining_orders = [o for o in orders if o != -1]
                    if remaining_orders:
                        min_order = min(remaining_orders)
                        #order_mapping = {o:i for i,o in enumerate(sorted(set(remaining_orders)))}
                        remaining_orders = [o for o in sample['order'] if o != -1]
                        order_mapping = {orig: new for new, orig in enumerate(remaining_orders)}
                        new_orders = [order_mapping.get(o.item(), -1) if o != -1 else -1 for o in orders]
                        orders = torch.LongTensor(new_orders)
                
                augmented.samples.append({
                    'coords': coords,
                    'order': orders,
                })
        
        return augmented

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_PANELS, scale=1.0):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term) * scale
        pe[:, 1::2] = torch.cos(position * div_term) * scale
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class MangaTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS):
        super().__init__()
        self.d_model = d_model
        
        # Enhanced input features
        self.input_proj = nn.Sequential(
            nn.Linear(10, d_model),  # x1,y1,x2,y2,xc,yc,w,h + area + aspect
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding with reduced scale
        self.pos_encoder = PositionalEncoding(d_model, scale=0.1)
        
        # Transformer
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # Output head
        self.order_head = nn.Linear(d_model, 1)
    
    def forward(self, x, mask=None):
        # Add geometric features
        # areas = x[:,:,2] * x[:,:,3]
        # aspect = x[:,:,2] / (x[:,:,3] + 1e-6)
        areas = x[:,:,6] * x[:,:,7]  # width * height
        aspect = x[:,:,6] / (x[:,:,7] + 1e-6)  # width/height
        
        #x = torch.cat([x, areas.unsqueeze(-1), aspect.unsqueeze(-1)], -1)
        x = torch.cat([x, areas.unsqueeze(-1), aspect.unsqueeze(-1)], -1)
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.order_head(x.transpose(0, 1)).squeeze(-1)

def train_model():
    # Load dataset
    dataset = MangaDataset(r'F:\AI\Yolo\samples2')
    dataset = dataset + dataset.create_augmented_copy()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=manga_collate_fn)
    
    # Initialize model
    model = MangaTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    train_taus = []
    epochs_list = []

    def old_ranking_loss(scores, targets, mask):
        """Pairwise ranking loss with correct ordering"""
        batch_loss = 0
        for i in range(scores.shape[0]):
            valid = ~mask[i]
            if valid.sum() < 2: continue
                
            s = scores[i][valid]
            t = targets[i][valid]
            
            # Create all pairs
            pair_diff = s.unsqueeze(1) - s.unsqueeze(0)
            target_diff = t.unsqueeze(1) - t.unsqueeze(0)
            
            # Loss for correct ordering (target_diff > 0 means panel j should come after i)
            loss = F.binary_cross_entropy_with_logits(
                pair_diff,
                (target_diff > 0).float(),  # Critical fix: changed from < to >
                reduction='mean'
            )
            batch_loss += loss
        
        return batch_loss / scores.shape[0]
    
    def ranking_loss(scores, targets, mask):
        """Pairwise ranking loss with correct ordering"""
        device = scores.device
        batch_loss = torch.tensor(0.0, requires_grad=True, device=device)
        total_valid = 0
        

        for i in range(scores.shape[0]):
            valid = ~mask[i]
            if valid.sum() < 2:
                continue  # Skip samples with less than 2 valid panels

            s = scores[i][valid]
            t = targets[i][valid]

            # Create all pairs
            pair_diff = s.unsqueeze(1) - s.unsqueeze(0)
            target_diff = t.unsqueeze(1) - t.unsqueeze(0)

            # Loss for correct ordering (target_diff > 0 means panel j should come after i)
            loss = F.binary_cross_entropy_with_logits(
                pair_diff,
                (target_diff > 0).float(),
                reduction='mean'
            )
            batch_loss = batch_loss + loss
            total_valid += 1

        if total_valid == 0:
            return batch_loss  # Return 0.0 tensor with gradient support
        return batch_loss / total_valid  # Normalize by valid samples count

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            coords = batch['coords'].to(DEVICE)
            targets = batch['order'].to(DEVICE)
            mask = (targets == -1)
            
            optimizer.zero_grad()
            scores = model(coords, mask)
            loss = ranking_loss(scores, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        tau = evaluate(model, train_loader)
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Tau={tau:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'manga_transformerv3v2_epoch{epoch+1}lr1e4.pth')
        train_taus.append(tau)
        epochs_list.append(epoch + 1)
    hyperparam_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
    plt.figure(figsize=(12, 5))
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Plot Tau
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_taus, label='Kendall\'s Tau', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Tau')
    plt.title('Kendall\'s Tau over Epochs')
    plt.legend()
    plt.figtext(0.95, 0.5, hyperparam_text, 
            bbox=dict(facecolor='white', alpha=0.5),
            verticalalignment='center',
            horizontalalignment='left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the plot
    plt.savefig(f'training_metrics_.png',bbox_inches='tight')
    return model

def evaluate(model, loader):
    model.eval()
    taus = []
    with torch.no_grad():
        for batch in loader:
            coords = batch['coords'].to(DEVICE)
            targets = batch['order'].to(DEVICE)
            mask = (targets == -1)
            
            scores = model(coords, mask)
            preds = torch.argsort(scores, dim=1)
            
            for i in range(preds.shape[0]):
                valid = ~mask[i]
                if valid.sum() > 1:
                    t = targets[i][valid].cpu().numpy()
                    p = preds[i][valid].cpu().numpy()
                    tau = kendalltau(t, p).correlation
                    if not np.isnan(tau):
                        taus.append(tau)
    
    return np.mean(taus) if taus else 0

if __name__ == "__main__":
    model = train_model()
    torch.save(model.state_dict(), 'manga_transformerv3v2lr1e4.pth')
