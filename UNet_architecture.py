import torch
import torch.nn as nn
import torch.nn.functional as F

class ExponentialMovingAverage:
    """
    Implements Exponential Moving Average (EMA) for updating model parameters.
    """
    def __init__(self, decay_rate):
        self.decay_rate = decay_rate
        self.steps_taken = 0

    def update_parameters(self, averaged_model, current_model):
        """
        Updates the parameters of the EMA model with the current model's parameters.
        """
        for param_current, param_avg in zip(current_model.parameters(), averaged_model.parameters()):
            param_avg.data = self.compute_average(param_avg.data, param_current.data)

    def compute_average(self, old_param, new_param):
        """
        Computes the weighted average of old and new parameters based on the decay rate.
        """
        if old_param is None:
            return new_param
        return self.decay_rate * old_param + (1 - self.decay_rate) * new_param

    def apply_ema(self, ema_model, model, start_step=2000):
        """
        Applies EMA if the specified step count is reached; otherwise, resets parameters.
        """
        if self.steps_taken < start_step:
            self.initialize_parameters(ema_model, model)
        else:
            self.update_parameters(ema_model, model)
        self.steps_taken += 1

    def initialize_parameters(self, ema_model, model):
        """
        Initializes the EMA model's parameters with the current model's parameters.
        """
        ema_model.load_state_dict(model.state_dict())


class AttentionModule(nn.Module):
    """
    Implements a self-attention mechanism for processing feature maps.
    """
    def __init__(self, feature_dim, spatial_size):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm([feature_dim])
        self.feedforward = nn.Sequential(
            nn.LayerNorm([feature_dim]),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, inputs):
        """
        Processes the input feature maps using self-attention and a feedforward network.
        """
        reshaped_input = inputs.view(-1, self.feature_dim, self.spatial_size * self.spatial_size).transpose(1, 2)
        normed_input = self.norm(reshaped_input)
        attention_out, _ = self.attention(normed_input, normed_input, normed_input)
        residual = attention_out + reshaped_input
        output = self.feedforward(residual) + residual
        return output.transpose(2, 1).view(-1, self.feature_dim, self.spatial_size, self.spatial_size)


class ResidualConvBlock(nn.Module):
    """
    Implements a double convolution block with an optional residual connection.
    """
    def __init__(self, input_channels, output_channels, hidden_channels=None, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        hidden_channels = hidden_channels if hidden_channels else output_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, output_channels),
        )

    def forward(self, x):
        """
        Applies the double convolution, with or without a residual connection.
        """
        if self.use_residual:
            return F.gelu(x + self.conv_block(x))
        return self.conv_block(x)


class DownSampleBlock(nn.Module):
    """
    Implements a downsampling block for U-Net, incorporating convolution and embedding layers.
    """
    def __init__(self, input_channels, output_channels, embedding_dim=256):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ResidualConvBlock(input_channels, input_channels, use_residual=True),
            ResidualConvBlock(input_channels, output_channels),
        )
        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, output_channels),
        )

    def forward(self, x, time_embedding):
        """
        Downsamples the input and adds a learned time-dependent embedding.
        """
        x = self.downsample(x)
        time_emb = self.embedding_layer(time_embedding)[:, :, None, None]
        return x + time_emb.expand_as(x)


class UpSampleBlock(nn.Module):
    """
    Implements an upsampling block for U-Net, incorporating convolution and embedding layers.
    """
    def __init__(self, input_channels, output_channels, embedding_dim=256):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_block = nn.Sequential(
            ResidualConvBlock(input_channels, input_channels, use_residual=True),
            ResidualConvBlock(input_channels, output_channels, input_channels // 2),
        )
        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, output_channels),
        )

    def forward(self, x, skip_connection, time_embedding):
        """
        Upsamples the input, concatenates with skip connection, and applies convolution and embedding.
        """
        x = self.upsample(x)
        x = torch.cat([skip_connection, x], dim=1)
        x = self.conv_block(x)
        time_emb = self.embedding_layer(time_embedding)[:, :, None, None]
        return x + time_emb.expand_as(x)


class UNetModel(nn.Module):
    """
    Implements a U-Net architecture with self-attention and time embeddings for image-to-image tasks.
    """
    def __init__(self, input_channels=3, output_channels=3, time_embedding_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_embedding_dim = time_embedding_dim
        self.input_conv = ResidualConvBlock(input_channels, 64)
        self.down1 = DownSampleBlock(64, 128)
        self.attn1 = AttentionModule(128, 32)
        self.down2 = DownSampleBlock(128, 256)
        self.attn2 = AttentionModule(256, 16)
        self.down3 = DownSampleBlock(256, 256)
        self.attn3 = AttentionModule(256, 8)

        self.bottleneck = nn.Sequential(
            ResidualConvBlock(256, 512),
            ResidualConvBlock(512, 512),
            ResidualConvBlock(512, 256),
        )

        self.up1 = UpSampleBlock(512, 128)
        self.attn4 = AttentionModule(128, 16)
        self.up2 = UpSampleBlock(256, 64)
        self.attn5 = AttentionModule(64, 32)
        self.up3 = UpSampleBlock(128, 64)
        self.attn6 = AttentionModule(64, 64)
        self.output_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def generate_positional_embedding(self, time, embedding_dim):
        """
        Generates sinusoidal positional embeddings for the given time steps.
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_dim, 2, device=self.device).float() / embedding_dim))
        pos_enc_a = torch.sin(time * inv_freq)
        pos_enc_b = torch.cos(time * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, time):
        """
        Processes the input through the U-Net with positional embeddings.
        """
        time = self.generate_positional_embedding(time.unsqueeze(-1).float(), self.time_embedding_dim)

        enc1 = self.input_conv(x)
        enc2 = self.attn1(self.down1(enc1, time))
        enc3 = self.attn2(self.down2(enc2, time))
        enc4 = self.attn3(self.down3(enc3, time))

        bottleneck = self.bottleneck(enc4)

        dec1 = self.attn4(self.up1(bottleneck, enc3, time))
        dec2 = self.attn5(self.up2(dec1, enc2, time))
        dec3 = self.attn6(self.up3(dec2, enc1, time))

        return self.output_conv(dec3)