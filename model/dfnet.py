import timm
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
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
            lr: float = 5e-4,
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


class DFModule(nn.Module):

    def __init__(
            self,
            in_chans,
            embed_dim=256,
            depth=2,
            num_heads=8,
            mlp_ratio=4,
            num_patches=4,
            dropout=0,
    ):
        super().__init__()

        # ======================
        # STN: localization net
        # ======================
        self.loc_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 6, 1)
        )

        # init as identity transform
        self.loc_net[-1].weight.data.zero_()
        self.loc_net[-1].bias.data.zero_()
        self.loc_net[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=1
        )
        self.pe = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.ln = nn.LayerNorm(embed_dim)

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
            nn.Conv2d(embed_dim, in_chans, kernel_size=1),
            nn.InstanceNorm2d(in_chans),
            nn.Sigmoid()
        )

        self.w1_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, in_chans, kernel_size=1),
            nn.Sigmoid()
        )
        self.chan_pro = nn.Sequential(
            nn.Conv2d(in_chans * 2, in_chans, kernel_size=1),
            nn.BatchNorm2d(in_chans),
            nn.SiLU(),
            nn.Conv2d(in_chans, in_chans, kernel_size=1),
        )

        self.alpha_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chans, in_chans, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_chans, 1, kernel_size=1),
            nn.Sigmoid()
        )


    # ======================
    # STN warp function
    # ======================
    def spatial_transform(self, x, theta):
        B, C, H, W = x.shape

        theta = theta.view(B, 2, 3)

        grid = F.affine_grid(
            theta,
            size=x.size(),
            align_corners=False
        )

        x = F.grid_sample(
            x,
            grid,
            align_corners=False
        )

        return x

    def forward(self, x):

        B, C, H, W = x.shape

        # ======================
        # 1. STN alignment
        # ======================
        res = x
        # ======================
        # 2. token attention
        # ======================
        x = self.patch_embed(x)
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.ln(x_flat) #+ self.pe
        x_enc = self.encoder(x_flat)

        # w = self.w_fc(x_enc)
        x_enc = x_enc.view(B, H, W, -1).permute(0, 3, 1, 2)
        w = self.w_fc(x_enc)
        # w_spa = self.wsp_fc(x_enc)
        # w = w / (w.mean(dim=(2, 3), keepdim=True) + 1e-6)

        # ======================
        # 3. foreground / background
        # ======================
        front = res * w
        # back = res * (1 - w)

        front_2d = front
        # back_2d = back

        # info_full = res.var(dim=(2, 3), keepdim=True).mean(dim=1, keepdim=True)
        # info_focus = front.var(dim=(2, 3), keepdim=True).mean(dim=1, keepdim=True)

        # information gain
        # delta = info_focus - info_full
        # theta = self.loc_net(w.detach())
        # front_2d = self.spatial_transform(front_2d, theta)
        # front_2d = front_2d * alpha + (1 - alpha) * res
        # front_2d =torch.cat((self.chan_pro(res), self.chan_pro(front_2d)), dim=1)
        # front_2d = torch.cat((res, front_2d), dim=1)
        # front_2d = self.chan_pro(front_2d)
        return front_2d, torch.mean(w, dim=1, keepdim=True)



class DFViT(nn.Module):

    def __init__(
        self,
        num_classes=5,
    ):
        super().__init__()

        self.backbone = EfficientNetV2Backbone(num_classes=num_classes)

        # DF blocks at semantic stages
        self.df1 = DFModule(in_chans=64, embed_dim=128, num_heads=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.df2 = DFModule(in_chans=160, embed_dim=256, num_patches=1024)
        self.bn2 = nn.BatchNorm2d(160)
        self.df3 = DFModule(in_chans=256, embed_dim=512, num_patches=256)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.conv_pro_stem = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.SiLU()
        )
        self.conv_pro_upper = nn.Sequential(
            nn.Conv2d(160, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.SiLU()
        )
        self.conv_pro_middel = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.SiLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 3, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes),
        )
        self.register_buffer("proto_normal", torch.zeros(512 * 3))
        self.register_buffer("proto_init", torch.tensor(0))
        self.proto_momentum = 0.99
        self.threshold = nn.Parameter(
            torch.tensor([
                0.5,
                1.5,
                2.5,
                3.5
            ])
        )

    def extract(self, x, need_fr_bk=False):
        x = self.backbone.stem(x)
        x = self.backbone.stage1(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        res = x
        x, w = self.df1(x)
        x = x + res
        x = self.bn1(x)
        df_x1 = x
        # df_x1 = F.adaptive_avg_pool2d(df_x1, 1).flatten(1)
        x = self.backbone.stage4(x)
        x = self.backbone.stage5(x)
        res = x
        x, w = self.df2(x)
        x = x + res
        x = self.bn2(x)
        df_x2 = x
        df_x2_pool = F.adaptive_avg_pool2d(df_x2, 1).flatten(1)
        x = self.backbone.stage6(x)
        # hidden = F.adaptive_avg_pool2d(x, 1).flatten(1)
        # x, _ = self.df3(x)
        # x = self.bn3(x)
        # df_x3 = x
        # df_x3_pool = F.adaptive_avg_pool2d(df_x3, 1).flatten(1)
        df_x1 = self.conv_pro_stem(df_x1)
        df_x1 = self.pool(df_x1)
        df_x2 = self.conv_pro_upper(df_x2)
        df_x2 = self.pool(df_x2)
        df_x3 = self.conv_pro_middel(x)
        df_x3 = self.pool(df_x3)
        x = torch.cat((df_x1, df_x2, df_x3), dim=1)
        if need_fr_bk:
            return x, [x.squeeze()], w
        return x

    def score_to_onehot(
            self,
            score
    ):
        # B×1
        score = score.unsqueeze(-1)

        # B×4
        exceed = (
                score >
                self.threshold
        )

        cls = exceed.sum(
            dim=-1
        )

        onehot = F.one_hot(
            cls,
            num_classes=5
        ).float()

        return onehot

    @torch.no_grad()
    def predict(self, x):
        self.eval()

        feat = self.extract(x).flatten(1)
        # B
        pred = self.fc(
            feat
        )

        return pred

    def bk_front_contrast(
            self,
            bk_area,
            front_area,
            tau=0.2,
    ):

        bk = F.normalize(
            bk_area.squeeze(1),
            dim=-1
        )

        front = F.normalize(
            front_area,
            dim=-1
        )

        # token重要性
        score = (
                front *
                bk.unsqueeze(1)
        ).sum(-1)

        weight = torch.softmax(
            score,
            dim=1
        )

        front_proto = (
                weight.unsqueeze(-1)
                *
                front
        ).sum(1)

        front_proto = F.normalize(
            front_proto,
            dim=-1
        )

        sim = (
                bk *
                front_proto
        ).sum(-1)

        loss = torch.exp(
            sim / tau
        )

        return loss.mean()

    @staticmethod
    def disease_normal_loss(
            feat,
            y,
            proto_normal,
            margin=0.3,
            eps=1e-6
    ):

        feat = F.normalize(feat, dim=-1, eps=eps)
        proto = F.normalize(proto_normal, dim=-1, eps=eps)

        loss = feat.new_tensor(0.)

        normal_mask = (y == 0)
        disease_mask = (y > 0)

        # ========== normal compact ==========
        if normal_mask.any():
            sim_n = torch.sum(feat[normal_mask] * proto, dim=-1)
            loss = loss + (1 - sim_n).mean()

        # ========== disease separation ==========
        if disease_mask.any():
            sim_d = torch.sum(feat[disease_mask] * proto, dim=-1)
            loss = loss + F.relu(sim_d - margin).mean()

        return loss

    def update_proto(self, feat, y):
        with torch.no_grad():

            normal = feat[y == 0]

            if normal.shape[0] == 0:
                return

            batch_proto = normal.mean(dim=0)

            if self.proto_init == 0:
                self.proto_normal.copy_(batch_proto)
                self.proto_init.fill_(1)
            else:
                self.proto_normal.mul_(self.proto_momentum)
                self.proto_normal.add_(batch_proto * (1 - self.proto_momentum))

    @staticmethod
    def ranking_loss(
            severity,
            label,
            margin=0.5
    ):
        score_diff = (
                severity[:, None]
                -
                severity[None, :]
        )

        label_diff = (
                label[:, None]
                -
                label[None, :]
        ).float()

        mask = label_diff > 0

        if not mask.any():
            return severity.new_tensor(0.)

        target = (
                margin
                *
                label_diff.abs()
        )

        loss = F.relu(
            target
            -
            score_diff
        )

        return loss[mask].mean()

    @staticmethod
    def active_loss(ap, y):
        """
        ap:
            任意形状预测激活值（建议范围[0,1]）

        y:
            标签，0表示正常，>0表示疾病
        """

        target = (y > 0).float()

        # 保证shape一致
        target = target.view_as(ap)

        loss = F.mse_loss(
            ap.float(),
            target
        )

        return loss

    def forward(self, x, label):

        self.train()

        front_area_2d, df_feats, w = self.extract(x, need_fr_bk=True)
        # loss_focus = self.bk_front_contrast(
        #     front_area_2d.reshape(front_area_2d.shape[0], front_area_2d.shape[1], -1).transpose(1, 2),
        #     bk_area_2.reshape(bk_area_2.shape[0], bk_area_2.shape[1], -1).transpose(1, 2),
        # )
        # logits_mix = self.backbone.stage2(mix_area_2d)
        # front_area_2d = self.backbone.stage7(front_area_2d)
        x = front_area_2d.flatten(1)
        logits = self.fc(x)# .squeeze(-1)

        # ====================
        # 2. Probability
        # ====================
        prob = torch.softmax(
            logits,
            dim=-1
        )

        # ====================
        # 3. Expectation
        # ====================
        bins = torch.arange(
            self.num_classes,
            device=logits.device,
            dtype=prob.dtype
        )

        score = (
                prob *
                bins[None]
        ).sum(-1)

        # loss_cls = F.cross_entropy(logits, label) #+ F.cross_entropy(bk_logits, torch.zeros_like(label))
        loss_dis = 0
        for df_feat in df_feats:
            self.update_proto(df_feat, label)
            loss_dis += self.disease_normal_loss(
                df_feat,
                label,
                self.proto_normal
            )
            # loss_dis += self.disease_normal_loss(df_feat, label)

        loss_active = 0
        loss_cls = 0.1 * F.smooth_l1_loss(
            score,
            label.float()
        ) + F.cross_entropy(logits, label)
        loss = (
            loss_cls # + 0.1 * loss_rank
            + 0 * loss_dis
            + 0 * loss_active
            # + 0 * loss_sparse
        )

        train_info = {
            "loss": {'total': loss.item(),
                     "cls": loss_cls.item(),
                     "dis": loss_dis.item(),
                     "act": loss_active
                     },
            "w": w.detach().cpu(),
        }

        return loss, train_info


class ConvNeXtBackbone(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            features_only=True,
            out_indices=(2,)   # stage3
        )

        self.out_channels = 192

    def forward(self, x):

        x = self.backbone(x)[0]

        # B x 192 x 14 x 14
        return x


class EfficientNetV2Backbone(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        model = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.DEFAULT
        )

        self.stem = model.features[0]

        self.stage1 = model.features[1:2]   # 24ch
        self.stage2 = model.features[2:3]   # 48ch
        self.stage3 = model.features[3:4]   # 64ch
        self.stage4 = model.features[4:5]   # 128ch
        self.stage5 = model.features[5:6]   # 160ch
        self.stage6 = model.features[6:7]    # 256ch
        self.stage7 = model.features[7:]  # 256ch

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(1280, num_classes)

    def forward_features(self, x):

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)

        return x

    def forward(self, x):

        x = self.forward_features(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)

        return x


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
        in_ch=128,
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
