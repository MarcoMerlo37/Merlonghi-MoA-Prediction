import torch.nn as nn
import torch.nn.functional as F

class MoAConv1D(nn.Module):
    """1D convolutional network for tabular data.
    The input vector is expanded by a dense layer into a pseudo-sequence of
    shape (n_channels, seq_len), over which 1D convolutions slide. Unlike the
    MLP/ResNet, which mix all inputs in every neuron, the convolutions share
    weights across neighbouring positions of the learned representation and
    are followed by pooling: a different inductive bias, in the spirit of the
    1D-CNNs used in the MoA competition."""

    name = "cnn1d"

    def __init__(self, input_dim, output_dim=206, n_channels=256, seq_len=16,
                 pooled_len=8, dropout_rates=(0.2, 0.3)):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len

        self.batch_norm_input = nn.BatchNorm1d(input_dim)
        self.dropout_input = nn.Dropout(dropout_rates[0])
        self.expand = nn.Linear(input_dim, n_channels * seq_len)

        # (B, 256, 16) -> conv k=5 -> pool -> (B, 256, 8)
        self.conv_block1 = nn.Sequential(
            nn.BatchNorm1d(n_channels),
            nn.Dropout(dropout_rates[1]),
            nn.Conv1d(n_channels, n_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(pooled_len),
        )
        # (B, 256, 8) -> conv k=3 -> (B, 256, 8)
        self.conv_block2 = nn.Sequential(
            nn.Dropout(dropout_rates[1]),
            nn.Conv1d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(n_channels * pooled_len, output_dim),
        )

    def forward(self, x):
        x = self.dropout_input(self.batch_norm_input(x))
        x = F.silu(self.expand(x))
        x = x.view(x.size(0), self.n_channels, self.seq_len)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.head(x)