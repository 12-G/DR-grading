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
