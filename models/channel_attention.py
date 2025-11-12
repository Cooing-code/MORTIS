import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class ChannelAttention(nn.Module):

    def __init__(self, n_channels, d_model, dropout=0.1):
        super(ChannelAttention, self).__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.learnable_fusion_weights = nn.Parameter(torch.randn(n_channels))
        expert_weights = torch.ones(n_channels) / n_channels
        self.register_buffer('expert_fusion_weights', expert_weights)
        self.fusion_gate_alpha = nn.Parameter(torch.tensor(0.0))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, channel_outputs):
        if isinstance(channel_outputs, list):
            try:
                channel_outputs = torch.stack(channel_outputs, dim=0)
            except Exception as e:
                valid_outputs = [t for t in channel_outputs if isinstance(t, torch.Tensor) and t.ndim == 3]
                if not valid_outputs:
                    raise ValueError('Channel outputs list is empty or contains invalid tensors.') from e
                first_shape = valid_outputs[0].shape
                if not all((t.shape == first_shape for t in valid_outputs)):
                    raise ValueError(f'Channel outputs have inconsistent shapes. Found shapes like {[t.shape for t in valid_outputs]}') from e
                if len(valid_outputs) != self.n_channels:
                    pass
                channel_outputs = torch.stack(valid_outputs, dim=0)
        n_channels, batch_size, seq_len, d_model = channel_outputs.shape
        if n_channels != self.n_channels:
            raise ValueError(f"Input channel count ({n_channels}) does not match module's n_channels ({self.n_channels})")
        alpha = torch.sigmoid(self.fusion_gate_alpha)
        normalized_learnable_w = F.softmax(self.learnable_fusion_weights, dim=0)
        mixed_weights = alpha * normalized_learnable_w + (1 - alpha) * self.expert_fusion_weights
        final_fusion_weights = mixed_weights / (mixed_weights.sum() + 1e-08)
        channel_outputs = channel_outputs.permute(1, 2, 0, 3)
        fusion_weights_reshaped = final_fusion_weights.view(1, 1, self.n_channels, 1)
        weighted_outputs = channel_outputs * fusion_weights_reshaped
        fused_output = weighted_outputs.sum(dim=2)
        fused_output = self.dropout(fused_output)
        fused_output = self.out_proj(fused_output)
        return (fused_output, final_fusion_weights.detach())

class CrossChannelAttention(nn.Module):

    def __init__(self, n_channels, d_model, n_heads=1, dropout=0.1):
        super(CrossChannelAttention, self).__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / sqrt(self.d_head)

    def forward(self, channel_outputs):
        if isinstance(channel_outputs, list):
            channel_outputs = torch.stack(channel_outputs, dim=0)
        n_channels, batch_size, seq_len, d_model = channel_outputs.shape
        x = channel_outputs.permute(1, 2, 0, 3)
        x_flat = x.reshape(batch_size * seq_len, n_channels, d_model)
        q = self.query_proj(x_flat).reshape(batch_size * seq_len, n_channels, self.n_heads, self.d_head)
        k = self.key_proj(x_flat).reshape(batch_size * seq_len, n_channels, self.n_heads, self.d_head)
        v = self.value_proj(x_flat).reshape(batch_size * seq_len, n_channels, self.n_heads, self.d_head)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k)
        scores = scores * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)
        context = context.permute(0, 2, 1, 3).reshape(batch_size * seq_len, n_channels, d_model)
        context = self.out_proj(context)
        enhanced_outputs = context.reshape(batch_size, seq_len, n_channels, d_model)
        enhanced_outputs = enhanced_outputs.permute(2, 0, 1, 3)
        avg_attention = attention_weights.mean(dim=1)
        reshaped_attention = avg_attention.reshape(batch_size, seq_len, n_channels, n_channels)
        final_attention = reshaped_attention.mean(dim=1)
        return (enhanced_outputs, final_attention)
