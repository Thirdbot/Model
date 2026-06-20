import torch
import torch.nn as nn

class MaskDecoder(nn.Module):
    def __init__(self, hidden_size, width_size=100,height_size=507):
        super().__init__()
        self.width_size = width_size
        self.height_size = height_size
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, width_size * height_size),
        )

    def forward(self, seg_hidden):
        logits = self.net(seg_hidden)
        return logits.view(seg_hidden.size(0), 1, self.height_size, self.width_size)
