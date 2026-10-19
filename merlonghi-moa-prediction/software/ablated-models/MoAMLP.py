import torch.nn as nn

class MoAMLP(nn.Module):
    name = "mlp"

    def __init__(self, input_dim, output_dim=206,
                 hidden_units=(1024, 1024, 512),
                 dropout_rates=(0.2, 0.3, 0.3)):
        super().__init__()
        self.batch_norm_input = nn.BatchNorm1d(input_dim)
        self.dropout_input = nn.Dropout(dropout_rates[0])

        self.dense1 = nn.Linear(input_dim, hidden_units[0])
        self.batch_norm1 = nn.BatchNorm1d(hidden_units[0])
        self.dropout1 = nn.Dropout(dropout_rates[1])

        self.dense2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.batch_norm2 = nn.BatchNorm1d(hidden_units[1])
        self.dropout2 = nn.Dropout(dropout_rates[2])

        self.dense3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.batch_norm3 = nn.BatchNorm1d(hidden_units[2])
        self.dropout3 = nn.Dropout(dropout_rates[2])

        self.output = nn.Linear(hidden_units[2], output_dim)

    def forward(self, x):
        x = self.dropout_input(self.batch_norm_input(x))

        x = F.silu(self.dense1(x))
        x = self.dropout1(self.batch_norm1(x))

        x = F.silu(self.dense2(x))
        x = self.dropout2(self.batch_norm2(x))

        x = F.silu(self.dense3(x))
        x = self.dropout3(self.batch_norm3(x))

        return self.output(x)