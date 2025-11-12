import torch
import torch.nn as nn
import numpy as np
import math

class AbsolutePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def rotary_apply_coords(x, coords):
    if coords.shape[-1] != x.shape[-1]:
        seq_len = coords.shape[0]
        d_model = x.shape[-1]
        position = torch.arange(0, seq_len, device=x.device, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * (-math.log(10000.0) / d_model))
        freqs = torch.einsum('i , j -> i j', position, div_term)
        coords = torch.cat((freqs, freqs), dim=-1)
    if x.ndim == 4:
        coords = coords.unsqueeze(1).repeat(1, x.shape[2], 1)
        coords = coords.unsqueeze(0)
    else:
        coords = coords.unsqueeze(0)
    x = x * coords.cos() + rotary_rotate_half(x) * coords.sin()
    return x

def rotary_rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape(x.shape)

class RotaryPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, base=10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self._initialised = False
        if d_model is not None:
            self._build_cache(max_len, torch.device('cpu'))

    def _build_cache(self, seq_len, device):
        if self.d_model is None:
            raise ValueError('d_model must be set before building cache.')
        assert self.d_model % 2 == 0
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.d_model, 2, device=device).float() / self.d_model)
        self.register_buffer('inv_freq', inv_freq.detach(), persistent=False)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        coords = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('coords_cache', coords.detach(), persistent=False)
        self._initialised = True

    def _apply_rotary(self, x):
        seq_len = x.shape[1]
        device = x.device
        if not self._initialised or seq_len > self.coords_cache.shape[0] or self.coords_cache.device != device:
            self._build_cache(max(seq_len, self.max_len), device)
        coords = self.coords_cache[:seq_len]
        return rotary_apply_coords(x, coords)

    def forward(self, q, k):
        return (self._apply_rotary(q), self._apply_rotary(k))

class RWaveCenteredPositionalEncoding(nn.Module):

    def __init__(self, d_model, r_peak_detector=None, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.r_peak_detector = r_peak_detector
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, r_peak_indices=None):
        batch_size, seq_len, _ = x.shape
        if r_peak_indices is None and self.r_peak_detector is not None:
            r_peak_indices = []
            for i in range(batch_size):
                signal = x[i, :, 0].cpu().numpy()
                try:
                    r_peaks = self.r_peak_detector(signal)
                    r_peak_indices.append(r_peaks)
                except Exception as e:
                    r_peak_indices.append([])
        if r_peak_indices is None or all((len(peaks) == 0 for peaks in r_peak_indices)):
            return self.pe[:, :seq_len, :].expand(batch_size, -1, -1)
        pos_encodings = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        for i in range(batch_size):
            r_peaks = r_peak_indices[i]
            if not isinstance(r_peaks, (list, np.ndarray)) or len(r_peaks) == 0:
                pos_encodings[i] = self.pe[0, :seq_len, :]
                continue
            r_peaks = np.array(r_peaks)
            sample_pos_encoding = torch.zeros(seq_len, self.d_model, device=x.device)
            if len(r_peaks) > 1:
                rr_intervals = np.diff(r_peaks)
                first_rr = rr_intervals[0] if len(rr_intervals) > 0 else self.max_len // 2
                rr_intervals = np.insert(rr_intervals, 0, first_rr)
            elif len(r_peaks) == 1:
                rr_intervals = np.array([self.max_len // 2])
            else:
                pos_encodings[i] = self.pe[0, :seq_len, :]
                continue
            avg_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else 1.0
            avg_rr = max(1.0, avg_rr)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
            r_peaks_tensor = torch.tensor(r_peaks, device=x.device, dtype=torch.float)
            dist = torch.abs(positions - r_peaks_tensor)
            nearest_r_indices = torch.argmin(dist, dim=1)
            nearest_r_pos = r_peaks_tensor[nearest_r_indices]
            rr_intervals_tensor = torch.tensor(rr_intervals, device=x.device, dtype=torch.float)
            nearest_rr = rr_intervals_tensor[nearest_r_indices]
            nearest_rr = torch.clamp(nearest_rr, min=1.0)
            rel_pos = (positions.squeeze() - nearest_r_pos) / nearest_rr
            cardiac_phase = torch.abs(rel_pos) % 1.0
            phase_pos_expanded = cardiac_phase.unsqueeze(1)
            div_term_gpu = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
            sample_pos_encoding[:, 0::2] = torch.sin(phase_pos_expanded * div_term_gpu)
            sample_pos_encoding[:, 1::2] = torch.cos(phase_pos_expanded * div_term_gpu)
            if self.d_model >= 8:
                sample_pos_encoding[:, 0] = torch.sin(rel_pos * math.pi)
                sample_pos_encoding[:, 1] = torch.cos(rel_pos * math.pi)
                sample_pos_encoding[:, 2] = torch.sin(cardiac_phase * 2 * math.pi)
                sample_pos_encoding[:, 3] = torch.cos(cardiac_phase * 2 * math.pi)
                sample_pos_encoding[:, 4] = torch.clamp(torch.abs(positions.squeeze() - nearest_r_pos) / nearest_rr, max=1.0)
                is_r_peak = torch.isin(positions.squeeze(), r_peaks_tensor)
                sample_pos_encoding[:, 5] = is_r_peak.float()
                norm_rr = torch.clamp(nearest_rr / avg_rr, min=0.5, max=2.0)
                sample_pos_encoding[:, 6] = norm_rr
                hrv_feature = torch.zeros(seq_len, device=x.device)
                valid_hrv_indices = nearest_r_indices > 0
                if torch.any(valid_hrv_indices):
                    prev_rr = rr_intervals_tensor[nearest_r_indices[valid_hrv_indices] - 1]
                    hrv = torch.abs(nearest_rr[valid_hrv_indices] - prev_rr) / avg_rr
                    hrv_feature[valid_hrv_indices] = torch.clamp(hrv, max=1.0)
                sample_pos_encoding[:, 7] = hrv_feature
            pos_encodings[i] = sample_pos_encoding
        return pos_encodings
