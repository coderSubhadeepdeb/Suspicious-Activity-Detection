import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell with GroupNorm on the gate outputs for training stability.
    GroupNorm works better than BatchNorm inside RNNs because it normalises
    within a single sample (no dependence on batch size or sequence position).
    """

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        # GroupNorm: normalise across channels within each sample.
        # num_groups=8 works well when hidden_channels ≥ 32.
        num_groups = min(8, hidden_channels)
        self.norm_i = nn.GroupNorm(num_groups, hidden_channels)
        self.norm_f = nn.GroupNorm(num_groups, hidden_channels)
        self.norm_o = nn.GroupNorm(num_groups, hidden_channels)
        self.norm_g = nn.GroupNorm(num_groups, hidden_channels)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates    = self.gates(combined)

        in_gate, forget_gate, out_gate, cell_gate = gates.chunk(4, dim=1)

        # Normalise before activation — improves gradient flow
        in_gate    = torch.sigmoid(self.norm_i(in_gate))
        forget_gate = torch.sigmoid(self.norm_f(forget_gate))
        out_gate   = torch.sigmoid(self.norm_o(out_gate))
        cell_gate  = torch.tanh(self.norm_g(cell_gate))

        c_next = forget_gate * c + in_gate * cell_gate
        h_next = out_gate * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers     = num_layers
        self.hidden_channels = hidden_channels

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))

    def forward(self, x):
        # x: (batch, seq_len, channels, height, width)
        batch, seq_len, _, height, width = x.size()
        device = x.device

        h, c = [], []
        for cell in self.cells:
            h_i, c_i = cell.init_hidden(batch, height, width, device)
            h.append(h_i)
            c.append(c_i)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(x_t, h[i], c[i])
                x_t = h[i]
            outputs.append(h[-1].unsqueeze(1))

        return torch.cat(outputs, dim=1)