import torch
import torch.nn as nn


class MaskDecoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        width_size=100,
        height_size=507,
        feature_channels=64,
        feature_height=32,
        feature_width=8,
    ):
        super().__init__()
        self.width_size = width_size
        self.height_size = height_size
        self.feature_channels = feature_channels
        self.feature_height = feature_height
        self.feature_width = feature_width

        self.fallback_image_features = nn.Parameter(
            torch.randn(1, feature_channels, feature_height, feature_width) * 0.02
        )

        self.query = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, feature_channels),
        )

        self.image_projection = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_channels // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feature_channels // 2, feature_channels // 4, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_channels // 4),
            nn.GELU(),
            nn.Conv2d(feature_channels // 4, 1, kernel_size=1),
        )

    def forward(self, seg_hidden, image_features=None):
        if image_features is None:
            image_features = self.fallback_image_features.expand(
                seg_hidden.size(0),
                -1,
                -1,
                -1,
            )
        else:
            image_features = self._match_feature_channels(image_features)
            image_features = self.image_projection(image_features)

        query = torch.sigmoid(self.query(seg_hidden)).unsqueeze(-1).unsqueeze(-1)
        conditioned = image_features * query

        logits = self.decoder(conditioned)
        return torch.nn.functional.interpolate(
            logits,
            size=(self.height_size, self.width_size),
            mode="bilinear",
            align_corners=False,
        )

    def _match_feature_channels(self, image_features):
        channels = image_features.shape[1]
        if channels == self.feature_channels:
            return image_features

        if channels > self.feature_channels:
            groups = torch.tensor_split(
                image_features,
                self.feature_channels,
                dim=1,
            )
            return torch.cat(
                [group.mean(dim=1, keepdim=True) for group in groups],
                dim=1,
            )

        pad = self.feature_channels - channels
        return torch.nn.functional.pad(image_features, (0, 0, 0, 0, 0, pad))
