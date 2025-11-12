import torch
import torch.nn as nn
import torch.nn.functional as F

class FANLayer(nn.Module):

    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation='gelu', with_gate=False):
        super(FANLayer, self).__init__()
        assert 0 < p_ratio < 0.5, 'p_ratio must be between 0 and 0.5'
        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2
        self.input_linear_p = nn.Linear(input_dim, p_output_dim)
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation if activation else lambda x: x
        self.with_gate = with_gate
        if with_gate:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        if self.with_gate:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate * torch.cos(p), gate * torch.sin(p), (1 - gate) * g), dim=-1)
        else:
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        return output
