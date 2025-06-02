import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 5

class PositionalEncoding(nn.Module):
    """Generates positional encodings for transformer inputs"""
    def __init__(self, d_model, max_len=20, scale=1.0):
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
    """Transformer model for manga reading order prediction"""
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS):
        super().__init__()
        self.d_model = d_model
        
        self.input_proj = nn.Sequential(
            nn.Linear(10, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, scale=0.1)
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.order_head = nn.Linear(d_model, 1)

    @staticmethod
    def preprocess_panel(xc, yc, w, h):
        """Convert YOLO format panel to model input format"""
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        return [x1, y1, x2, y2, xc, yc, w, h]

    def forward(self, x, mask=None):
        areas = x[:,:,6] * x[:,:,7]
        aspect = x[:,:,6] / (x[:,:,7] + 1e-6)
        x = torch.cat([x, areas.unsqueeze(-1), aspect.unsqueeze(-1)], -1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.order_head(x.transpose(0, 1)).squeeze(-1)

    def predict_sequence(self, panels):
        """Predict reading order from preprocessed panels"""
        padded = panels + [[0]*8]*(20 - len(panels))
        tensor = torch.FloatTensor(padded).unsqueeze(0).to(next(self.parameters()).device)
        
        with torch.no_grad():
            scores = self(tensor)
        
        scores = scores.cpu().numpy().flatten()
        num_real = len(panels)
        predicted_order = []
        
        for idx in np.argsort(-scores):
            if idx < num_real and idx not in predicted_order:
                predicted_order.append(idx)
            if len(predicted_order) == num_real:
                break
        return predicted_order

class MangaVisualizer:
    """Handles visualization of manga panels and reading order"""
    @staticmethod
    def visualize_sequence(panels, sequence, img_size=1000, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)
        ax.invert_yaxis()
        ax.set_title("Manga Panel Reading Order", fontsize=14)

        for idx, panel in enumerate(panels):
            x1, y1, x2, y2, xc, yc, w, h = panel
            if w == 0 and h == 0:
                continue

            x1 *= img_size
            y1 *= img_size
            width = (x2 - x1) * img_size
            height = (y2 - y1) * img_size

            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

            if idx in sequence:
                order = sequence.index(idx)
                ax.text(
                    x1 + 5, y1 + 15, f"{order + 1}",
                    fontsize=12, color='blue', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                )

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

class SequencerTransformer():
    def __init__(self, model_path):
        self.model = MangaTransformer().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
        self.model.eval()
    
    # Expects xc yc w h
    def preprocessor(self, panels):
        preprocessed = [MangaTransformer.preprocess_panel(*panel) for panel in panels]
        return preprocessed
    
    def predict(self, panels):
        preprocessed = self.preprocessor(panels)
        sequence = self.model.predict_sequence(preprocessed)
        return sequence
    
