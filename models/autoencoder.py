import torch
import torch.nn as nn
from models.convlstm import ConvLSTM


class SpatialEncoder(nn.Module):
    def __init__(self, channels):
        super(SpatialEncoder, self).__init__()
        layers = []
        in_ch = 1  # grayscale input
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class SpatialDecoder(nn.Module):
    def __init__(self, channels):
        super(SpatialDecoder, self).__init__()
        layers = []
        channels = list(reversed(channels))
        for i in range(len(channels) - 1):
            layers += [
                nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True)
            ]
        layers += [
            nn.ConvTranspose2d(channels[-1], 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        ]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class ConvLSTMAE(nn.Module):
    def __init__(self, encoder_channels, convlstm_hidden, sequence_length):
        super(ConvLSTMAE, self).__init__()
        self.sequence_length = sequence_length

        self.spatial_encoder = SpatialEncoder(encoder_channels)
        self.spatial_decoder = SpatialDecoder(encoder_channels)

        self.convlstm = ConvLSTM(
            input_channels=encoder_channels[-1],
            hidden_channels=convlstm_hidden,
            kernel_size=3,
            num_layers=2
        )

        self.bottleneck = nn.Conv2d(convlstm_hidden, encoder_channels[-1], kernel_size=1)

    def forward(self, x):
        # x: (batch, seq_len, 1, H, W)
        batch, seq_len, c, h, w = x.size()

        # Encode each frame spatially
        encoded_frames = []
        for t in range(seq_len):
            enc = self.spatial_encoder(x[:, t])
            encoded_frames.append(enc.unsqueeze(1))
        encoded = torch.cat(encoded_frames, dim=1)

        # Process sequence with ConvLSTM
        lstm_out = self.convlstm(encoded)

        # Decode each frame
        reconstructed = []
        for t in range(seq_len):
            feat = self.bottleneck(lstm_out[:, t])
            dec = self.spatial_decoder(feat)
            reconstructed.append(dec.unsqueeze(1))

        return torch.cat(reconstructed, dim=1)