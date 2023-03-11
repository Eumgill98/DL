import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(kwargs['input_shape'], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, kwargs['input_shape'])
        )
    
    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder

       

