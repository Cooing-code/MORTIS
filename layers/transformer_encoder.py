import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.FANLayer import FANLayer
from layers.self_attention import AttentionLayer, DSAttention, FullAttention, ProbAttention
from layers.r_wave_positional_encoding import RWaveCenteredPositionalEncoding, RotaryPositionalEncoding

class ConvLayer(nn.Module):

    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=2, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class EncoderLayer(nn.Module):

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu', use_fan_layer=False, fan_p_ratio=0.25, fan_activation='gelu', fan_with_gate=False):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.use_fan_layer = use_fan_layer
        if use_fan_layer:
            self.ff_layer = FANLayer(input_dim=d_model, output_dim=d_model, p_ratio=fan_p_ratio, activation=fan_activation, with_gate=fan_with_gate)
            self.conv1 = None
            self.conv2 = None
            self.activation_fn = None
        else:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
            if isinstance(activation, str):
                self.activation_fn = getattr(F, activation)
            else:
                self.activation_fn = activation
            self.ff_layer = None

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        if self.use_fan_layer:
            y = self.ff_layer(y)
        else:
            if self.conv1 is None or self.conv2 is None or self.activation_fn is None:
                raise RuntimeError('Standard FFN layers not initialized when use_fan_layer=False')
            y = y.transpose(-1, 1)
            y = self.dropout(self.activation_fn(self.conv1(y)))
            y = self.dropout(self.conv2(y))
            y = y.transpose(-1, 1)
        x = x + self.dropout(y)
        x = self.norm2(x)
        return (x, attn)

class Encoder(nn.Module):

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, positional_encoding_type='absolute', d_model=None, max_len=5000):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.positional_encoding_type = positional_encoding_type
        self.pos_embedding = None
        if self.positional_encoding_type == 'absolute':
            if d_model is None:
                raise ValueError('d_model must be provided for absolute positional encoding')
            self.pos_embedding = nn.Embedding(max_len, d_model)
            nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)
        elif self.positional_encoding_type == 'rwave_centered':
            if d_model is None:
                raise ValueError('d_model must be provided for R-Wave positional encoding')
            self.pos_embedding = RWaveCenteredPositionalEncoding(d_model=d_model, max_len=max_len)
        elif self.positional_encoding_type == 'rotary':
            pass
        elif self.positional_encoding_type == 'none':
            pass
        else:
            raise ValueError(f'Unknown positional_encoding_type: {self.positional_encoding_type}')

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        B, L, D = x.shape
        if self.positional_encoding_type == 'absolute' and self.pos_embedding is not None:
            position_ids = torch.arange(L, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, -1)
            pos_embed = self.pos_embedding(position_ids)
            x = x + pos_embed
        elif self.positional_encoding_type == 'rwave_centered' and self.pos_embedding is not None:
            x = self.pos_embedding(x)
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
            if self.conv_layers is not None and i < len(self.conv_layers):
                x = self.conv_layers[i](x)
        if self.norm is not None:
            x = self.norm(x)
        return (x, attns)
