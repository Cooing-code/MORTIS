import torch

class TriangularCausalMask:

    def __init__(self, B, L, device='cpu'):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask:

    def __init__(self, B, H, L, index, scores, device='cpu'):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device)
        _mask_ex = _mask.unsqueeze(0).unsqueeze(0)
        _mask = _mask_ex.expand(B, H, L, scores.shape[-1])
        self._mask = _mask.clone()
        for i in range(B):
            for j in range(H):
                self._mask[i, j, index[i, j]] = False

    @property
    def mask(self):
        return self._mask
