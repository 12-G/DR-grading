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
        depth=1,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        num_classes=5,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim)
        )

        self.pos_drop = nn.Dropout(dropout)

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

        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self.opt = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3
        )

    def extract(self, x):

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        feat = self.encoder(x)

        w = torch.sigmoid(
            self.w_fc(feat)
        )

        return feat, w

    @torch.no_grad()
    def predict(self, x):

        self.eval()

        feat, w = self.extract(x)

        front = (feat * w).sum(1) / (w.sum(1) + 1e-6)

        logits = self.cls_head(front)

        return logits

    def bk_front_contrast(
            self,
            bk_area,
            front_area,
            tau=0.1
    ):
        B, C, D = front_area.shape

        # B D
        bk = F.normalize(
            bk_area.squeeze(1),
            dim=-1
        )

        # B C D
        front = F.normalize(
            front_area,
            dim=-1
        )

        # --------------------
        # 正样本：背景-背景
        # B×B
        # --------------------
        pos = bk @ bk.T

        eye = torch.eye(
            B,
            device=bk.device,
            dtype=torch.bool
        )

        pos = pos.masked_fill(
            eye,
            -1e9
        )

        pos = torch.logsumexp(
            pos / tau,
            dim=1
        )

        # --------------------
        # 负样本：背景-所有前景
        # B×B×C
        # --------------------
        neg = torch.einsum(
            "bd,ncd->bnc",
            bk,
            front
        )

        neg = neg.reshape(
            B,
            B * C
        )

        neg = torch.logsumexp(
            neg / tau,
            dim=1
        )

        # --------------------
        # InfoNCE
        # --------------------
        loss = -(pos - torch.logaddexp(
            pos,
            neg
        ))

        return loss.mean()

    def forward(self, x, label):

        self.train()

        feat, w = self.extract(x)

        bk_area = ((1 - w) * feat).mean(1, keepdim=True)

        front_area = w * feat

        d_focus_loss = self.bk_front_contrast(
            bk_area,
            front_area
        )

        front_feat = (
            front_area.sum(1)
            / (w.sum(1) + 1e-6)
        )

        logits = self.cls_head(
            front_feat
        )

        loss_cls = F.cross_entropy(
            logits,
            label
        )

        loss = (
            loss_cls
            + 0.2 * d_focus_loss
        )

        self.opt.zero_grad()

        loss.backward()

        self.opt.step()

        train_info = {
            "loss": loss.item(),
            "cls": loss_cls.item(),
            "focus": d_focus_loss.item(),
            "fg_ratio": w.mean().item()
        }

        return logits, train_info

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
