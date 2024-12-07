import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, text_dim=768, visual_dim=2048, hidden_dim=512):
        super().__init__()
        
        # Projection layers
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, text_features, visual_features):
        """
        Compute cross-modal attention between text and visual features
        """
        # Project features to common space
        text_proj = self.text_projection(text_features)
        visual_proj = self.visual_projection(visual_features)
        
        # Compute attention
        attn_output, attn_weights = self.attention(
            query=text_proj,
            key=visual_proj,
            value=visual_proj
        )
        
        return attn_output, attn_weights 