import torch
import torch.nn as nn
from models.convlstm import ConvLSTM


class SpatialEncoder(nn.Module):
    """
    Encodes a single (grayscale) frame into a compact latent map.
    Returns both the final encoded feature AND intermediate skip features
    (one per downsampling stage, excluding the last) for U-Net connections.
    """

    def __init__(self, channels):
        super(SpatialEncoder, self).__init__()
        self.layers = nn.ModuleList()
        in_ch = 1  # grayscale
        for out_ch in channels:
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            in_ch = out_ch

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Collect all but the last (bottleneck) activation as skip features
            if i < len(self.layers) - 1:
                skips.append(x)
        return x, skips   # (bottleneck_feat, [skip1, skip2, ...])


class SpatialDecoder(nn.Module):
    """
    U-Net decoder: each upsample stage receives the skip connection from the
    corresponding encoder stage via channel concatenation.

    With ENCODER_CHANNELS = [32, 64, 128]:
      - Bottleneck output: 128 ch  (from bottleneck conv)
      - Stage 1: upsample 128→64, cat skip_64   → 128 ch input
      - Stage 2: upsample 128→32, cat skip_32   → 64  ch input
      - Stage 3: upsample 64→1   (final output)
    """

    def __init__(self, channels):
        super(SpatialDecoder, self).__init__()
        rev = list(reversed(channels))   # e.g. [128, 64, 32]

        # ── Upsample + merge blocks ──────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        for i in range(len(rev) - 1):
            in_ch  = rev[i]
            out_ch = rev[i + 1]
            # After concat with skip the input will be doubled
            merged_ch = out_ch * 2   # skip has out_ch channels
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            # 1×1 conv to merge concatenated channels back to out_ch
            self.up_blocks.append(nn.Sequential(
                nn.Conv2d(merged_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))

        # ── Final head: upsample to original resolution ───────────────────────
        self.final = nn.Sequential(
            nn.ConvTranspose2d(rev[-1], 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x, skips):
        """
        x     : bottleneck feature  (batch, C_bottleneck, H, W)
        skips : list of skip tensors in encoder order [early→late] → reverse
        """
        rev_skips = list(reversed(skips))   # match decoder stages
        block_idx = 0
        for i in range(len(rev_skips)):
            up_block   = self.up_blocks[block_idx]
            merge_block = self.up_blocks[block_idx + 1]
            block_idx  += 2

            x = up_block(x)                    # upsample
            x = self.dropout(x)
            x = torch.cat([x, rev_skips[i]], dim=1)   # U-Net concat
            x = merge_block(x)                 # reduce channels

        return self.final(x)


class ConvLSTMAE(nn.Module):
    """
    Full spatio-temporal autoencoder:
      SpatialEncoder (per frame) → ConvLSTM → Bottleneck → SpatialDecoder (per frame)
    Skip connections from the spatial encoder flow into the decoder for each frame,
    preserving fine-grained spatial detail.
    """

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

        # Map ConvLSTM output back to encoder bottleneck size
        self.bottleneck = nn.Sequential(
            nn.Conv2d(convlstm_hidden, encoder_channels[-1], kernel_size=1),
            nn.BatchNorm2d(encoder_channels[-1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (batch, seq_len, 1, H, W)
        batch, seq_len, c, h, w = x.size()

        # ── Spatial encode every frame ───────────────────────────────────────
        encoded_frames = []
        frame_skips    = []   # per-frame skip connection lists
        for t in range(seq_len):
            enc, skips = self.spatial_encoder(x[:, t])
            encoded_frames.append(enc.unsqueeze(1))
            frame_skips.append(skips)
        encoded = torch.cat(encoded_frames, dim=1)   # (B, T, C, H', W')

        # ── Temporal processing ──────────────────────────────────────────────
        lstm_out = self.convlstm(encoded)            # (B, T, hidden, H', W')

        # ── Spatial decode every frame (with its own skip connections) ───────
        reconstructed = []
        for t in range(seq_len):
            feat = self.bottleneck(lstm_out[:, t])
            dec  = self.spatial_decoder(feat, frame_skips[t])
            reconstructed.append(dec.unsqueeze(1))

        return torch.cat(reconstructed, dim=1)       # (B, T, 1, H, W)