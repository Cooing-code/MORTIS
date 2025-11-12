import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePatchEmbedding(nn.Module):

    def __init__(self, d_model, patch_len, stride, padding, dropout=0.1):
        super(AdaptivePatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.projection = nn.Linear(patch_len, d_model)
        self.norm = nn.LayerNorm(patch_len)

    def forward(self, x):
        batch_size, seq_len, n_vars = x.shape
        if seq_len < self.patch_len:
            padding_needed = self.patch_len - seq_len
            x = F.pad(x, (0, 0, 0, padding_needed), mode='replicate')
            seq_len = x.shape[1]
        num_patch = (seq_len + 2 * self.padding - (self.patch_len - 1) - 1) // self.stride + 1
        x = x.permute(0, 2, 1)
        x_unfolded = F.unfold(x.unsqueeze(-1), kernel_size=(self.patch_len, 1), stride=(self.stride, 1), padding=(self.padding, 0))
        x_patched = x_unfolded.transpose(1, 2).reshape(batch_size, num_patch, n_vars, self.patch_len)
        channel_embeddings = []
        for i in range(n_vars):
            channel_patches = x_patched[:, :, i, :]
            channel_patches = self.norm(channel_patches)
            channel_embed = self.projection(channel_patches)
            channel_embed = self.dropout(channel_embed)
            channel_embeddings.append(channel_embed)
        return (channel_embeddings, n_vars)
