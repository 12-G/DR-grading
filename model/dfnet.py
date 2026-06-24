import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig


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
        backbone_chans=48,
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

        self.backbone = MobilenetV3Backbone()

        self.patch_embed = nn.Conv2d(
            backbone_chans,
            embed_dim,
            kernel_size=1,
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
            nn.LayerNorm([num_patches, embed_dim // 2]),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.LayerNorm([num_patches, 1]),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=backbone_chans, kernel_size=1),
            MobileNetV3Classifier(num_classes=num_classes)
        )

        self.opt = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3
        )

    def extract(self, x, need_fr_bk=False):
        x = self.backbone(x)
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        feat = self.encoder(x)

        w = torch.sigmoid(
            self.w_fc(feat)
        )
        front_area = w * feat
        front_area_2d = front_area.transpose(1, 2).reshape(B, C, H, W)
        if need_fr_bk:
            bk_area = ((1 - w) * feat).mean(1, keepdim=True)
            return front_area_2d, front_area, bk_area, w

        return front_area_2d

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        front = self.extract(x)
        logits = self.classifier(front)
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

    @staticmethod
    def sparse_loss(w):
        eps = 1e-8

        entropy = (
                -w * torch.log(w + eps)
                - (1 - w) * torch.log(1 - w + eps)
        )

        return entropy.mean()

    def forward(self, x, label):

        self.train()

        front_area_2d, front_area, bk_area, w = self.extract(x, need_fr_bk=True)
        loss_focus = self.bk_front_contrast(
            bk_area,
            front_area
        )

        logits = self.classifier(front_area_2d)

        loss_cls = F.cross_entropy(
            logits,
            label
        )

        loss_sparse = self.sparse_loss(w)

        loss = (
            loss_cls
            + 0 * loss_focus
            + 0 * loss_sparse
        )

        self.opt.zero_grad()

        loss.backward()

        self.opt.step()

        train_info = {
            "loss": {'total': loss.item(),
                     "cls": loss_cls.item(),
                     "focus": loss_focus.item(),
                     "sparse": loss_sparse.item()},
            "w": w.detach().cpu(),
        }

        return logits, train_info


class MobilenetV3Backbone(nn.Module):
    def __init__(self):
        super(MobilenetV3Backbone, self).__init__()
        model = models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )
        self.backbone = model.features[:8]

    def forward(self, x):
        x = self.backbone(x)
        return x


class MobileNetV3Classifier(nn.Module):

    def __init__(
        self,
        in_ch=48,
        num_classes=10,
        last_ch=576,
        drop=0.2
    ):
        super().__init__()

        self.blocks = nn.Sequential(

            # 48×14×14
            InvertedResidual(
                InvertedResidualConfig(
                    input_channels=in_ch,
                    kernel=5,
                    expanded_channels=288,
                    out_channels=96,
                    use_se=True,
                    activation="HS",
                    stride=2,
                    dilation=1,
                    width_mult=1.0
                ),
                norm_layer=nn.BatchNorm2d
            ),

            # 96×7×7
            InvertedResidual(
                InvertedResidualConfig(
                    input_channels=96,
                    kernel=5,
                    expanded_channels=576,
                    out_channels=96,
                    use_se=True,
                    activation="HS",
                    stride=1,
                    dilation=1,
                    width_mult=1.0
                ),
                norm_layer=nn.BatchNorm2d
            ),

            nn.Conv2d(
                96,
                last_ch,
                1,
                bias=False
            ),

            nn.BatchNorm2d(last_ch),

            nn.Hardswish()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.cls = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                last_ch,
                1024
            ),

            nn.Hardswish(),

            nn.Dropout(drop),

            nn.Linear(
                1024,
                num_classes
            )
        )

    def forward(self, x):

        x = self.blocks(x)

        x = self.pool(x)

        x = self.cls(x)

        return x


class Backbone(nn.Module):
    def __init__(
            self,
            image_size=224,
    ):
        super().__init__()



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
