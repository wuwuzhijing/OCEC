from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def _load_weight_file(path: str):
    """Load a weight file in .pth, .pt, or .safetensors format."""
    if path.endswith(".safetensors"):
        try:
            import safetensors.torch
            return safetensors.torch.load_file(path, device="cpu")
        except ImportError:
            raise ImportError("safetensors is required to load .safetensors files. Install with: pip install safetensors")
    else:
        # .pth / .pt — try weights_only first, fall back if needed
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            print(f"  [warn] weights_only load failed for {path}, retrying with weights_only=False")
            return torch.load(path, map_location="cpu", weights_only=False)


class ArcFaceHead(nn.Module):
    """ArcFace: Additive Angular Margin Loss"""
    def __init__(self, embedding_dim, num_classes=2, s=30.0, s_val = 10.0, m=0.50):
        super().__init__()
        self.s = s
        self.s_val = s_val
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding, labels=None):
        emb = F.normalize(embedding, dim=1)
        w = F.normalize(self.weight, dim=1)
        if w.dtype != emb.dtype:
            w = w.to(emb.dtype)  # 避免 AMP 下 dtype 不一致
        logits = F.linear(emb, w)   # (B, 2)

        if labels is None:
            # 验证/推理时：返回不带 scale 的 logits，避免概率分布过于极端
            return logits * self.s_val # 不使用 scale，让 softmax 产生更合理的概率分布

        # 训练时：ArcFace: cos(θ + m)，然后乘以 scale
        theta = torch.acos(torch.clamp(logits, -1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)

        one_hot = F.one_hot(labels.long(), num_classes=2)
        logits = logits * (1 - one_hot) + target_logits * one_hot
        return logits * self.s

class CosFaceHead(nn.Module):
    """
    CosFace with margin warmup:
    
    m_effective = m * min(1, epoch / warmup_epochs)
    """
    def __init__(self, embedding_dim, num_classes=2, m=0.35, s=8.0, warmup_epochs=10, s_val=4.0):
        super().__init__()
        self.m = m
        self.s = s
        self.s_val = s_val  # used in eval
        self.warmup_epochs = warmup_epochs
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # running value for tensorboard/debug
        self.current_margin = 0.0

    def forward(self, embedding, labels=None, epoch=None):
        """
        If labels=None → eval mode → no margin applied (but classifier still used).
        """
        emb = F.normalize(embedding, dim=1)
        w = F.normalize(self.weight, dim=1)
        if w.dtype != emb.dtype:
            w = w.to(emb.dtype)  # 避免 AMP 下半精度/全精度混算 dtype 不一致

        logits = torch.matmul(emb, w.t())  # cosine similarity

        if labels is None:
            # eval: scale logits but DO NOT apply margin
            return logits * self.s_val

        # ---- margin warmup ----
        if epoch is not None and self.warmup_epochs > 0:
            warm_factor = min(1.0, float(epoch) / float(self.warmup_epochs))
            self.current_margin = self.m * warm_factor
        else:
            self.current_margin = self.m

        one_hot = F.one_hot(labels.long(), num_classes=logits.size(1)).float().to(logits.device)

        logits_margin = logits - one_hot * self.current_margin
        return logits_margin * self.s
    
@dataclass
class ModelConfig:
    """Lightweight CNN configuration for OCEC."""

    base_channels: int = 32
    num_blocks: int = 4
    dropout: float = 0.3
    arch_variant: str = "baseline"
    expansion: int = 4
    se_reduction: int = 8
    head_variant: str = "auto"
    token_mixer_grid: tuple[int, int] = (2, 3)
    token_mixer_layers: int = 2
    margin_method: str = "cosface"  # "none", "arcface", "cosface"
    pretrained_backbone: str = ""  # "" = custom; or "mobilenet_v3_small", "efficientnet_b0", "resnet18", "repvgg_b0", ...
    pretrained_weights_dir: str = ""  # local dir for pre-downloaded weights (avoids network)


class _SepConvBlock(nn.Module):
    """Depthwise separable convolution block with optional residual."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.use_residual:
            x = x + identity
        x = F.relu(x, inplace=True)
        return x


class _SqueezeExcite(nn.Module):
    """Squeeze-and-excitation attention module."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        reduced = max(1, channels // max(1, reduction))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


class _InvertedResidualSEBlock(nn.Module):
    """MobileNetV2-style inverted residual block with squeeze-and-excite."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        if expansion < 1:
            raise ValueError("Expansion factor must be >= 1.")
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expansion

        layers = []
        if expansion != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ]
        )
        if se_reduction > 0:
            layers.append(_SqueezeExcite(hidden_dim, se_reduction))
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block(x)
        if self.use_residual:
            out = out + identity
        out = self.out_act(out)
        return out


class _LayerNorm2d(nn.Module):
    """LayerNorm operating on channels-last tensors but returning channels-first.
    Optimized version using GroupNorm for better performance."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        # 使用 GroupNorm(1) 代替 LayerNorm + permute，性能更好
        # GroupNorm(num_groups=1) 等价于 LayerNorm，但支持 channels-first
        self.norm = nn.GroupNorm(1, num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 直接使用 GroupNorm，无需 permute
        return self.norm(x)


class _ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block with depthwise conv and channel MLP.
    Optimized version using 1x1 conv instead of Linear + permute."""

    def __init__(self, dim: int, layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _LayerNorm2d(dim)
        # 使用 1x1 conv 代替 Linear，避免 permute 操作
        self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)
        if layer_scale_init > 0:
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        # 直接使用 conv，无需 permute
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            # gamma 需要 reshape 以匹配空间维度
            x = x * self.gamma.view(1, -1, 1, 1)
        x = x + shortcut
        return x


class _ConvNeXtDownsample(nn.Module):
    """Downsample layer used between ConvNeXt stages."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("ConvNeXt downsample stride must be 1 or 2.")
        if stride == 1 and in_channels == out_channels:
            self.op = None
        else:
            kernel = 2 if stride == 2 else 1
            self.op = nn.Sequential(
                _LayerNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.op is None:
            return x
        return self.op(x)


class _MLPMixerBlock(nn.Module):
    """Lightweight MLP-Mixer block operating on a fixed token grid."""

    def __init__(
        self,
        num_tokens: int,
        channels: int,
        dropout: float = 0.0,
        token_expansion: int = 2,
        channel_expansion: int = 2,
    ) -> None:
        super().__init__()
        token_hidden = max(num_tokens, num_tokens * token_expansion)
        channel_hidden = max(channels, channels * channel_expansion)
        self.norm_tokens = nn.LayerNorm(channels)
        self.token_fc1 = nn.Linear(num_tokens, token_hidden)
        self.token_act = nn.GELU()
        self.token_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.token_fc2 = nn.Linear(token_hidden, num_tokens)
        self.norm_channels = nn.LayerNorm(channels)
        self.channel_fc1 = nn.Linear(channels, channel_hidden)
        self.channel_act = nn.GELU()
        self.channel_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.channel_fc2 = nn.Linear(channel_hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, tokens, channels)
        residual = x
        y = self.norm_tokens(x)
        y = y.transpose(1, 2)  # (batch, channels, tokens)
        y = self.token_fc1(y)
        y = self.token_act(y)
        y = self.token_dropout(y)
        y = self.token_fc2(y)
        y = self.token_dropout(y)
        y = y.transpose(1, 2)
        x = residual + y

        y = self.norm_channels(x)
        y = self.channel_fc1(y)
        y = self.channel_act(y)
        y = self.channel_dropout(y)
        y = self.channel_fc2(y)
        y = self.channel_dropout(y)
        return x + y


def _select_attention_heads(channels: int) -> int:
    for candidate in range(min(8, channels), 0, -1):
        if channels % candidate == 0:
            return candidate
    return 1


class _TokenMixerHead(nn.Module):
    """Applies Transformer or MLP-Mixer across pooled spatial tokens before classification."""

    def __init__(
        self,
        channels: int,
        dropout: float,
        mixer_type: str,
        grid: tuple[int, int],
        layers: int,
    ) -> None:
        super().__init__()
        if mixer_type not in {"transformer", "mlp_mixer"}:
            raise ValueError(f"Unsupported token mixer type: {mixer_type}")
        self.grid = grid
        self.channels = channels
        self.num_tokens = grid[0] * grid[1]
        self.mixer_type = mixer_type
        self.layers = max(1, int(layers))
        dropout = float(max(0.0, dropout))

        if mixer_type == "transformer":
            heads = _select_attention_heads(channels)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=heads,
                dim_feedforward=max(128, channels * 2),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.mixer = nn.TransformerEncoder(encoder_layer, num_layers=self.layers)
        else:
            blocks = [
                _MLPMixerBlock(self.num_tokens, channels, dropout=dropout)
                for _ in range(self.layers)
            ]
            self.mixer = nn.Sequential(*blocks)

        self.final_norm = nn.LayerNorm(channels)
        self.final_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, height, width)
        h, w = self.grid
        height, width = x.shape[-2], x.shape[-1]
        if height % h != 0 or width % w != 0:
            raise RuntimeError(
                f"Feature map {height}x{width} is not divisible by requested token grid {h}x{w}. "
                "Adjust --token_mixer_grid or image/architecture settings."
            )
        kernel_h = height // h
        kernel_w = width // w
        tokens = F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        tokens = tokens.flatten(2).transpose(1, 2)  # (batch, tokens, channels)
        tokens = self.mixer(tokens)
        pooled = tokens.mean(dim=1)
        pooled = self.final_norm(pooled)
        pooled = self.final_dropout(pooled)
        logits = self.fc(pooled)
        return logits


class OCEC(nn.Module):
    """Compact mouth state classifier that outputs logits."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.margin_method = self.config.margin_method
        print(f"margin_method: {self.margin_method}")
        base = self.config.base_channels
        num_blocks = max(1, self.config.num_blocks)
        variant = (self.config.arch_variant or "baseline").lower()
        head_variant = getattr(self.config, "head_variant", "auto")
        if head_variant:
            head_variant = head_variant.lower()
        if head_variant == "auto":
            if variant == "inverted_se":
                self._head_variant = "avgmax_mlp"
            elif variant == "convnext":
                self._head_variant = "transformer"
            else:
                self._head_variant = "avg"
        else:
            self._head_variant = head_variant
        self._token_grid = self._ensure_token_grid(getattr(self.config, "token_mixer_grid", (3, 2)))
        self._token_layers = int(max(1, getattr(self.config, "token_mixer_layers", 2)))

        pretrained = (self.config.pretrained_backbone or "").strip().lower()
        self._is_pretrained = bool(pretrained)

        if self._is_pretrained:
            # ── Pretrained backbone path ──
            self._variant = "pretrained"
            self.stem = nn.Identity()
            self.features, channels = self._build_pretrained_backbone(pretrained)
            self._feature_channels = channels

            # Force simple avg head for pretrained backbone
            self._head_variant = "avg"
            head_in_features = self._head_input_dim(channels)
            self.head = self._build_head(head_in_features)

            # Only init head/margin_head weights, NOT the pretrained backbone
            self._init_head_weights()
        else:
            # ── Custom backbone path (original) ──
            if variant == "baseline":
                stem_layers = [
                    nn.Conv2d(3, base, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base),
                    nn.ReLU(inplace=True),
                ]
            elif variant == "inverted_se":
                stem_layers = [
                    nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(base),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(base, base, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base),
                    nn.SiLU(inplace=True),
                ]
            elif variant == "convnext":
                stem_layers = [
                    nn.Conv2d(3, base, kernel_size=4, stride=4, padding=0, bias=True),
                    _LayerNorm2d(base),
                ]
            else:
                raise ValueError(f"Unsupported architecture variant: {self.config.arch_variant}")
            self._variant = variant
            self.stem = nn.Sequential(*stem_layers)

            channels = base
            blocks = []
            for idx in range(num_blocks):
                stride = 2 if idx % 2 == 0 and idx > 0 else 1
                next_channels = channels * (2 if stride == 2 else 1)
                blocks.append(self._make_block(channels, next_channels, stride))
                channels = next_channels

            self.features = nn.Sequential(*blocks)
            self._feature_channels = channels

            if self._head_variant in {"avg", "avgmax_mlp"}:
                head_in_features = self._head_input_dim(channels)
                self.head = self._build_head(head_in_features)
            elif self._head_variant in {"transformer", "mlp_mixer"}:
                self.head = _TokenMixerHead(
                    channels,
                    dropout=self.config.dropout,
                    mixer_type=self._head_variant,
                    grid=self._token_grid,
                    layers=self._token_layers,
                )
            else:
                raise ValueError(f"Unsupported head variant: {self._head_variant}")

            self._init_weights()

        # ── Margin head (shared) ──
        # Embedding dropout: applies regularization directly to the feature
        # vector before it enters the margin_head.  This is critical because
        # when margin_head is active, self.head is bypassed and its dropout
        # never affects the loss path.
        self.embedding_dropout = nn.Dropout(self.config.dropout) if self.config.dropout > 0 else nn.Identity()

        if self.config.margin_method in ["arcface", "cosface"]:
            emb_dim = self._feature_channels
            if self.config.margin_method == "arcface":
                self.margin_head = ArcFaceHead(embedding_dim=emb_dim, num_classes=2, m=0.2, s=8, s_val=4)
            elif self.config.margin_method == "cosface":
                self.margin_head = CosFaceHead(embedding_dim=emb_dim, num_classes=2, m=0.18, s=12, s_val=6)
        else:
            self.margin_head = None

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_pretrained_backbone(self, name: str):
        """Load a pretrained backbone and return (features_module, out_channels).

        Supports torchvision models (mobilenet_v3_small, efficientnet_b0, resnet18,
        resnet34) and timm models (repvgg_*, and any other timm model).

        Pre-downloaded weights can be placed in pretrained_weights_dir to skip network.
        """
        print(f"Loading pretrained backbone: {name}")

        weights_dir = (self.config.pretrained_weights_dir or "").strip()
        if weights_dir:
            print(f"  Looking for local weights in: {weights_dir}")

        # ── torchvision backbones ──
        _TV_MODELS = {
            "mobilenet_v3_small": 576,
            "efficientnet_b0": 1280,
            "resnet18": 512,
            "resnet34": 512,
        }

        if name in _TV_MODELS:
            from torchvision import models

            if name == "mobilenet_v3_small":
                model = models.mobilenet_v3_small(weights=None)  # no auto-download
                features = model.features
            elif name == "efficientnet_b0":
                model = models.efficientnet_b0(weights=None)
                features = model.features
            elif name in ("resnet18", "resnet34"):
                model_cls = models.resnet18 if name == "resnet18" else models.resnet34
                model = model_cls(weights=None)
                # Modified stem for small (64×64) eye ROI:
                #   7×7 stride=2 → 3×3 stride=1 (keep more spatial detail)
                #   maxpool → Identity (skip aggressive downsampling)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
                features = nn.Sequential(
                    model.conv1, model.bn1, model.relu, model.maxpool,
                    model.layer1, model.layer2, model.layer3, model.layer4,
                )
            out_channels = _TV_MODELS[name]

            # Load weights from local dir if provided; otherwise checkpoint state dict fills them
            if weights_dir:
                target = model if name in ("resnet18", "resnet34") else features
                self._load_weights_from_dir(target, name, weights_dir)
                print(f"  Loaded local weights for: {name}")
            else:
                print(f"  Created {name} without pretrained weights (will load from checkpoint)")

        else:
            # ── timm backbones (repvgg_*, etc.) ──
            try:
                import timm
            except ImportError:
                raise ImportError(
                    f"timm is required for backbone '{name}'. Install with: pip install timm"
                )

            if weights_dir:
                model = timm.create_model(name, pretrained=False, num_classes=0, global_pool='')
                self._load_weights_from_dir(model, name, weights_dir)
                print(f"  Loaded local weights for: {name}")
            else:
                # No local weights dir: create without pretrained.
                # (checkpoint state dict will fill weights via load_state_dict later)
                model = timm.create_model(name, pretrained=False, num_classes=0, global_pool='')
                print(f"  Created {name} without pretrained weights (will load from checkpoint)")

            features = model
            out_channels = model.num_features

        print(f"  Pretrained backbone loaded: {name} (output channels={out_channels})")
        return features, out_channels

    @staticmethod
    def _load_weights_from_dir(model: nn.Module, name: str, weights_dir: str) -> None:
        """Load pretrained weights from a local directory.

        Tries: {dir}/{name}.pth, {dir}/{name}.safetensors, {dir}/{name}.pt
        """
        import os
        candidates = [
            os.path.join(weights_dir, f"{name}.pth"),
            os.path.join(weights_dir, f"{name}.safetensors"),
            os.path.join(weights_dir, f"{name}.pt"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                print(f"  Found: {path}")
                state_dict = _load_weight_file(path)
                # Handle wrapped state dicts
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif isinstance(state_dict, dict) and "model" in state_dict:
                    state_dict = state_dict["model"]
                model.load_state_dict(state_dict, strict=False)
                return
        raise FileNotFoundError(
            f"Could not find weights for '{name}' in {weights_dir}. "
            f"Tried: {candidates}"
        )

    def _init_head_weights(self) -> None:
        """Initialize ONLY head and margin_head weights (skip pretrained backbone)."""
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, labels=None, return_embedding=False, training=None, epoch=None) -> torch.Tensor:
        """
        Args:
            x: 输入图像
            labels: 标签（用于训练时的 margin-based loss）
            return_embedding: 是否返回 embedding
            training: 是否在训练模式（None 时自动判断）
        """
        x = self.stem(x)
        x = self.features(x)

        if self._head_variant in {"transformer", "mlp_mixer"}:
            # transformer 输出前的 pooled token 作为 embedding
            h, w = self._token_grid
            height, width = x.shape[-2], x.shape[-1]
            kernel_h = height // h
            kernel_w = width // w
            tokens = torch.nn.functional.avg_pool2d(x, (kernel_h, kernel_w), (kernel_h, kernel_w))
            tokens = tokens.flatten(2).transpose(1, 2)  # (B, T, C)
            embedding = tokens.mean(dim=1)              # (B, C)
            logits = self.head(x)
        else:
            # CNN 分支
            embedding = self._pool_features(x)   # (B,C)
            logits = self.head(embedding)

        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        if self.margin_head is not None:
            # Determine if margin applies (training only)
            use_margin = (training is not None and training) or (training is None and self.training)

            if use_margin:
                # Training: apply embedding dropout, then margin warmup
                emb = self.embedding_dropout(embedding)
                logits_margin = self.margin_head(emb, labels=labels, epoch=epoch)
                if return_embedding:
                    return logits_margin, embedding
                return logits_margin
            else:
                # Eval/inference → No margin, no dropout, scaled logits
                logits_no_margin = self.margin_head(embedding, labels=None)
                if return_embedding:
                    return logits_no_margin, embedding
                return logits_no_margin

        # 不使用 margin
        if return_embedding:
            return logits, embedding
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(open) probability.

        When margin_head is active (CosFace/ArcFace), forward returns (B,2) logits
        and we must use softmax, not sigmoid.  P(open) = softmax(logits)[:, 1].
        """
        logits = self.forward(x)
        if self.margin_head is not None and logits.ndim == 2 and logits.size(1) == 2:
            return torch.softmax(logits, dim=1)[:, 1]
        return torch.sigmoid(logits)

    def _make_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        if self._variant == "baseline":
            return _SepConvBlock(in_channels, out_channels, stride=stride)
        if self._variant == "inverted_se":
            expansion = int(max(1, getattr(self.config, "expansion", 4)))
            se_reduction = int(max(0, getattr(self.config, "se_reduction", 8)))
            return _InvertedResidualSEBlock(
                in_channels,
                out_channels,
                stride=stride,
                expansion=expansion,
                se_reduction=se_reduction,
            )
        if self._variant == "convnext":
            modules = []
            if stride != 1 or in_channels != out_channels:
                modules.append(_ConvNeXtDownsample(in_channels, out_channels, stride))
            modules.append(_ConvNeXtBlock(out_channels))
            return nn.Sequential(*modules)
        raise ValueError(f"Unsupported architecture variant: {self._variant}")

    def _head_input_dim(self, feature_channels: int) -> int:
        if self._head_variant == "avg":
            return feature_channels
        if self._head_variant == "avgmax_mlp":
            return feature_channels * 2
        raise ValueError(f"Unsupported head variant: {self._head_variant}")

    def _build_head(self, in_features: int) -> nn.Module:
        dropout = float(max(0.0, self.config.dropout))
        if self._head_variant == "avg":
            return nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_features, 1),
            )
        if self._head_variant == "avgmax_mlp":
            hidden = max(16, in_features // 2)
            layers = [
                nn.BatchNorm1d(in_features),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_features, hidden),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, 1))
            return nn.Sequential(*layers)
        raise ValueError(f"Unsupported head variant: {self._head_variant}")

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        if self._head_variant == "avg":
            return x.mean(dim=(-2, -1))
        if self._head_variant == "avgmax_mlp":
            avg = x.mean(dim=(-2, -1))
            maxv = torch.amax(x, dim=(-2, -1))
            return torch.cat([avg, maxv], dim=1)
        raise ValueError(f"Pooling not defined for head variant: {self._head_variant}")

    def _ensure_token_grid(self, grid: Any) -> tuple[int, int]:
        if not isinstance(grid, (tuple, list)) or len(grid) != 2:
            return (3, 2)
        h, w = int(grid[0]), int(grid[1])
        if h <= 0 or w <= 0:
            return (3, 2)
        return (h, w)
