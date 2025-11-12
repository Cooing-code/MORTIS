import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, List, Tuple

class RWaveCenteredPatchEmbedding(nn.Module):

    def __init__(self, d_model: int, patch_len: int, stride: int=None, padding_patch: bool=False, r_peak_detector=None, adaptive_patch_len: bool=False, min_patch_len: int=32, max_patch_len: int=128):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride if stride is not None else patch_len // 2
        self.padding_patch = padding_patch
        self.r_peak_detector = r_peak_detector
        self.adaptive_patch_len = adaptive_patch_len
        self.min_patch_len = min_patch_len
        self.max_patch_len = max_patch_len
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Linear(2, d_model // 4)

    def forward(self, x, r_peak_indices=None):
        batch_size, seq_len, n_vars = x.shape
        device = x.device
        if r_peak_indices is None and self.r_peak_detector is not None:
            r_peak_indices = []
            for i in range(batch_size):
                signal = x[i, :, 0].cpu().numpy()
                r_peaks = self.r_peak_detector(signal)
                r_peak_indices.append(r_peaks)
        if r_peak_indices is None:
            return self._standard_patch_embedding(x)
        all_patches = []
        all_positions = []
        all_patch_lens = []
        max_patches = 0
        for i in range(batch_size):
            r_peaks = r_peak_indices[i]
            if len(r_peaks) == 0:
                patches, positions, patch_lens = self._standard_patch_embedding(x[i:i + 1])
                all_patches.append(patches.squeeze(0))
                all_positions.append(positions.squeeze(0))
                all_patch_lens.append(patch_lens)
                max_patches = max(max_patches, patches.shape[1])
                continue
            rr_intervals = np.diff(r_peaks, prepend=r_peaks[0], append=seq_len)
            patches = []
            positions = []
            patch_lens = []
            for j, r_pos in enumerate(r_peaks):
                if self.adaptive_patch_len:
                    prev_rr = rr_intervals[j]
                    next_rr = rr_intervals[j + 1] if j + 1 < len(rr_intervals) else prev_rr
                    patch_len = min(self.max_patch_len, max(self.min_patch_len, int((prev_rr + next_rr) / 2)))
                    if patch_len % 2 == 1:
                        patch_len += 1
                else:
                    patch_len = self.patch_len
                stride = self.stride
                window_positions = [r_pos]
                left_pos = r_pos - stride
                while left_pos >= 0:
                    window_positions.insert(0, left_pos)
                    left_pos -= stride
                right_pos = r_pos + stride
                while right_pos < seq_len:
                    window_positions.append(right_pos)
                    right_pos += stride
                for center_pos in window_positions:
                    half_len = patch_len // 2
                    start_idx = center_pos - half_len
                    end_idx = center_pos + half_len
                    if start_idx < 0 or end_idx > seq_len:
                        if not self.padding_patch:
                            continue
                        padded_patch = torch.zeros(patch_len, n_vars, device=device)
                        valid_start = max(0, start_idx)
                        valid_end = min(seq_len, end_idx)
                        patch_start = max(0, -start_idx)
                        patch_end = patch_len - max(0, end_idx - seq_len)
                        padded_patch[patch_start:patch_end, :] = x[i, valid_start:valid_end, :]
                        patch = padded_patch
                    else:
                        patch = x[i, start_idx:end_idx, :]
                    patch = patch.transpose(0, 1)
                    phase_features = self._calculate_cardiac_phase(r_peaks, center_pos)
                    phase_features = torch.tensor(phase_features, device=device)
                    channel_embeddings = []
                    for c in range(n_vars):
                        channel_patch = patch[c]
                        channel_emb = self.value_embedding(channel_patch)
                        channel_embeddings.append(channel_emb)
                    patch_emb = torch.stack(channel_embeddings).mean(dim=0)
                    pos_emb = self.position_embedding(phase_features)
                    combined_emb = torch.cat([patch_emb[:self.d_model - self.d_model // 4], pos_emb], dim=0)
                    patches.append(combined_emb)
                    positions.append(center_pos)
                    patch_lens.append(patch_len)
            if patches:
                patches = torch.stack(patches)
                positions = torch.tensor(positions, device=device)
                patch_lens = torch.tensor(patch_lens, device=device)
            else:
                patches = torch.zeros(1, self.d_model, device=device)
                positions = torch.tensor([0], device=device)
                patch_lens = torch.tensor([self.patch_len], device=device)
            all_patches.append(patches)
            all_positions.append(positions)
            all_patch_lens.append(patch_lens)
            max_patches = max(max_patches, patches.shape[0])
        padded_patches = []
        padded_positions = []
        padded_patch_lens = []
        for patches, positions, patch_lens in zip(all_patches, all_positions, all_patch_lens):
            n_patches = patches.shape[0]
            if n_patches < max_patches:
                padding = torch.zeros(max_patches - n_patches, self.d_model, device=device)
                patches = torch.cat([patches, padding], dim=0)
                pos_padding = torch.zeros(max_patches - n_patches, device=device)
                positions = torch.cat([positions, pos_padding], dim=0)
                len_padding = torch.zeros(max_patches - n_patches, device=device)
                patch_lens = torch.cat([patch_lens, len_padding], dim=0)
            padded_patches.append(patches)
            padded_positions.append(positions)
            padded_patch_lens.append(patch_lens)
        patches_out = torch.stack(padded_patches)
        positions_out = torch.stack(padded_positions)
        patch_lens_out = torch.stack(padded_patch_lens)
        return (patches_out, positions_out, patch_lens_out)

    def _standard_patch_embedding(self, x):
        batch_size, seq_len, n_vars = x.shape
        device = x.device
        n_patches = max(1, (seq_len - self.patch_len) // self.stride + 1)
        all_patches = torch.zeros(batch_size, n_patches, self.d_model, device=device)
        positions = torch.zeros(batch_size, n_patches, device=device)
        patch_lens = torch.ones(batch_size, n_patches, device=device) * self.patch_len
        for i in range(n_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            if end_idx > seq_len:
                break
            center_pos = start_idx + self.patch_len // 2
            positions[:, i] = center_pos
            patches = x[:, start_idx:end_idx, :]
            for b in range(batch_size):
                patch = patches[b].transpose(0, 1)
                channel_embeddings = []
                for c in range(n_vars):
                    channel_patch = patch[c]
                    channel_emb = self.value_embedding(channel_patch)
                    channel_embeddings.append(channel_emb)
                patch_emb = torch.stack(channel_embeddings).mean(dim=0)
                phase_features = torch.tensor([math.sin(math.pi), math.cos(math.pi)], device=device)
                pos_emb = self.position_embedding(phase_features)
                combined_emb = torch.cat([patch_emb[:self.d_model - self.d_model // 4], pos_emb], dim=0)
                all_patches[b, i] = combined_emb
        return (all_patches, positions, patch_lens)

    def _calculate_cardiac_phase(self, r_peaks, position):
        if len(r_peaks) == 0:
            return [0.0, -1.0]
        r_dists = [abs(r - position) for r in r_peaks]
        nearest_idx = np.argmin(r_dists)
        nearest_r = r_peaks[nearest_idx]
        if nearest_idx + 1 < len(r_peaks):
            rr_interval = r_peaks[nearest_idx + 1] - nearest_r
        elif nearest_idx > 0:
            rr_interval = nearest_r - r_peaks[nearest_idx - 1]
        else:
            rr_interval = 128
        rel_pos = (position - nearest_r) / max(1, rr_interval)
        cardiac_phase = (rel_pos + 1) % 1
        return [math.sin(cardiac_phase * 2 * math.pi), math.cos(cardiac_phase * 2 * math.pi)]
