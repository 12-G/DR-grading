import torch
from torch import nn
import torch.nn.functional as F


class TimmFeatureEncoder(nn.Module):
    """
    timm backbone + 分类头训练封装
    """

    def __init__(
            self,
            model_name: str,
            num_classes: int = 5,
            pretrained: bool = True,
            lr: float = 1e-3,
    ):
        super().__init__()

        import timm

        # 创建模型
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes
        )

        self.dim = int(getattr(self.backbone, "num_features", 0))
        if self.dim <= 0:
            raise RuntimeError(f"无法从 {model_name} 读取 num_features")

        # 优化器
        self.opt = torch.optim.Adam(
            self.parameters(),
            lr=lr
        )

        # loss
        self.ce_loss = nn.CrossEntropyLoss()

    @torch.no_grad()
    def predict(self, x):
        """
        推理
        """
        self.eval()

        logits = self.backbone(x)
        return logits

    def forward(self, img, label):
        """
        单步训练
        """
        self.train()

        self.opt.zero_grad()

        pred = self.backbone(img)

        loss = self.ce_loss(pred, label)

        loss.backward()

        self.opt.step()

        train_info = {'loss': loss}

        return pred, train_info


class DFViT(nn.Module):
    def __init__(
            self,
            image_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=512,
            depth=4,
            num_heads=8,
            mlp_ratio=4,
            dropout=0.1,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size

        num_patches = (image_size // patch_size) ** 2

        # ------------------
        # Patch Embedding
        # ------------------
        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                num_patches,
                embed_dim
            )
        )

        self.pos_drop = nn.Dropout(dropout)

        # ------------------
        # Transformer Encoder
        # ------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.w_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x):
        B = x.shape[0]
        # patchify
        x = self.patch_embed(x)
        # [B,D,H',W']

        x = x.flatten(2)
        x = x.transpose(1, 2)
        # [B,N,D]

        x = x + self.pos_embed

        x = self.pos_drop(x)

        x = self.encoder(x)
        w = self.w_fc(x)
        w = torch.sigmoid(w)

        bk_area = (1 - w) * x
        bk_area = torch.mean(bk_area, dim=1, keepdim=True)


        return cls_feat


class Backbone(nn.Module):

    def __init__(
            self,
            image_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=5,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            dropout=0.1,
            lr=1e-4,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size

        num_patches = (image_size // patch_size) ** 2

        # ------------------
        # Patch Embedding
        # ------------------
        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                num_patches + 1,
                embed_dim
            )
        )

        self.pos_drop = nn.Dropout(dropout)

        # ------------------
        # Transformer Encoder
        # ------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        # 分类头
        self.head = nn.Linear(
            embed_dim,
            num_classes
        )

        self.opt = torch.optim.AdamW(
            self.parameters(),
            lr=lr
        )

        self.ce_loss = nn.CrossEntropyLoss()

    def extract_feature(self, x):
        """
        x:
        [B,C,H,W]

        output:
        [B,D]
        """

        B = x.shape[0]

        # patchify
        x = self.patch_embed(x)
        # [B,D,H',W']

        x = x.flatten(2)
        x = x.transpose(1, 2)
        # [B,N,D]

        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat(
            [cls, x],
            dim=1
        )

        x = x + self.pos_embed

        x = self.pos_drop(x)

        x = self.encoder(x)

        cls_feat = x[:, 0]

        return cls_feat

    @torch.no_grad()
    def predict(self, x):
        self.eval()

        logits = self.forward(x)

        return logits.argmax(-1)

    def forward(self, img, label=None):
        feat = self.extract_feature(img)

        logits = self.head(feat)

        if label is None:
            return logits

        self.train()

        self.opt.zero_grad()

        loss = self.ce_loss(
            logits,
            label
        )

        loss.backward()

        self.opt.step()

        train_info = {
            "loss": loss.item()
        }

        return logits, train_info
