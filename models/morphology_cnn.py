import torch
import torch.nn as nn

class MorphologyCNN(nn.Module):

    def __init__(self, input_channels, output_dim=128, kernels=[3, 5, 7], num_filters=32):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        conv_output_size = 0
        for kernel_size in kernels:
            block = nn.Sequential(nn.Conv1d(input_channels, num_filters, kernel_size=kernel_size, padding='same'), nn.BatchNorm1d(num_filters), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
            self.conv_blocks.append(block)
            conv_output_size += num_filters
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(conv_output_size, output_dim)
        self.relu_fc = nn.ReLU()
        self._input_channels = input_channels

    def forward(self, x):
        if x.shape[1] == self._input_channels:
            pass
        elif x.shape[2] == self._input_channels:
            x = x.permute(0, 2, 1)
        else:
            raise ValueError(f'Input tensor shape {x.shape} does not match expected channel count {self._input_channels}. Expected [B, C, L] or [B, L, C].')
        cnn_outputs = []
        for block in self.conv_blocks:
            cnn_outputs.append(block(x))
        combined = torch.cat(cnn_outputs, dim=1)
        flat_features = self.flatten(combined)
        output = self.relu_fc(self.fc(flat_features))
        return output
