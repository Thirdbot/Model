import torch.nn as nn

class MaskDecoder:
    def __init__(self, hidden_size, output_size=MASK_OUTPUT_SIZE):
        super().__init__()
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size * output_size),
        )


    def forward(self, seg_hidden):
        logits = self.net(seg_hidden)
        return logits.view(seg_hidden.size(0), 1, self.output_size, self.output_size)

