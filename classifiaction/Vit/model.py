import torch
from torch import nn
from torch import Tensor
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


class Vit(nn.Module):
    def __init__(self, in_channel: int = 3, img_size:int = 224, 
                 patch_size: int = 16, emb_dim:int = 16*16*3, 
                 n_enc_layers:int = 15, num_heads:int = 3, 
                 forward_dim:int = 4, dropout_ratio: float = 0.2, 
                 n_classes:int = 1000):
        super().__init__()

        self.image_emb = image_embedding(in_channel, img_size, patch_size, emb_dim)
        self.transformer_encoders = nn.ModuleList([encoder_block(emb_dim, num_heads, forward_dim, dropout_ratio) for _ in range(n_enc_layers)])

        self.reduce_layer = Reduce('b n e -> b e', reduction='mean')
        self.normalization = nn.LayerNorm(emb_dim)
        self.classification_head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        x = self.image_emb(x)

        attentions = []
        for encoder in self.transformer_encoders:
            x, att = encoder(x)
            attentions.append(att)


        x = self.reduce_layer(x)
        x = self.normalization(x)
        x = self.classification_head(x)

        return x, attentions

class image_embedding(nn.Module):
    def __init__(self, in_channels: int = 3, img_size: int = 224,  patch_size: int = 16, emb_dim: int = 16*16*3):
        super().__init__()

        #이미지를 패치 size로 나누고 flatten
        self.path_flatten = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        #cls_token, position encodeing 
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1,  emb_dim))

    def forward(self, x):
        batch, _, _, _ = x.shape

        x = self.path_flatten(x)

        c = repeat(self.cls_token, '() n d -> b n d', b=batch)
        x = torch.cat((c, x), dim=1)
        x = x + self.positions

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim:int = 16*16*3, num_heads:int=8,  dropout_ratio: float = 0.2, **kwargs):
        super().__init__()

        self.emb_dim = emb_dim 
        self.num_heads = num_heads
        self.scaling = (self.emb_dim // num_heads)  ** -0.5

        self.value = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)
        self.att_drop = nn.Dropout(dropout_ratio)

        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = rearrange(Q, 'b q (h d) -> b h q d', h=self.num_heads)
        K = rearrange(K, 'b k (h d) -> b h d k', h=self.num_heads)
        V = rearrange(V, 'b v (h d) -> b h v d', h=self.num_heads)

        weight = torch.matmul(Q, K)
        weight = weight * self.scaling

        attention = torch.softmax(weight, dim=1)
        attention = self.att_drop(attention)

        context = torch.matmul(attention, V)
        context = rearrange(context, 'b h q d -> b q (h d)')

        x = self.linear(context)
        return x , attention
    

class MlpBlock(nn.Module):
    def __init__(self, emb_dim: int = 16*16*3, forward_dim: int = 4, dropout_ratio: float = 0.2, **kwargs):
        super().__init__()

        self.Mlp = nn.Sequential(
            nn.Linear(emb_dim, forward_dim * emb_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(forward_dim * emb_dim, emb_dim)
        )

    def forward(self, x):
        x = self.Mlp(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, emb_dim:int = 16*16*3, num_heads:int = 8, forward_dim: int = 4, dropout_ratio:float = 0.2):
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_dim)
        self.mha = MultiHeadAttention(emb_dim, num_heads, dropout_ratio)
        
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MlpBlock(emb_dim, forward_dim, dropout_ratio)
        self.residual_dropout = nn.Dropout(dropout_ratio)
    
    def forward(self, x):
        x_ = self.norm1(x)
        x_, attention = self.mha(x_)
        x = x_ + self.residual_dropout(x)

        x_ = self.norm2(x)
        x_ = self.mlp(x_)
        x = x_ + self.residual_dropout(x)

        return x, attention
