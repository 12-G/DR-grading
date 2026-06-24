import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, cohen_kappa_score
# -------------------------
# metrics
# -------------------------
from data.utils import make_loader
from model.dfnet import TimmFeatureEncoder, DFViT, Backbone


def compute_metrics(pred, label, average="macro"):
    """
    pred: (N,)
    label: (N,)
    """
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()

    acc = (pred == label).mean()
    f1 = f1_score(label, pred, average=average)
    qwk = cohen_kappa_score(label, pred, weights="quadratic")

    return {
        "acc": float(acc),
        "f1": float(f1),
        "qwk": float(qwk),
    }





class TrainManager:
    def __init__(
            self,
            model,
            train_loader,
            val_loader=None,
            test_loader=None,
            device="cuda",
            use_amp=True,
            save_dir="./checkpoints",
            print_step=10,
            vis_step=50
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.fig_save_dir = 'figs'
        os.makedirs(save_dir, exist_ok=True)

        self.print_step = print_step
        self.vis_step = vis_step

    def visualize_attention(
            self,
            img,
            w,
            step,
            max_show=4
    ):

        # img: B×3×H×W
        # w: B×H×W
        w = w.reshape((-1, 14, 14))
        B = min(max_show, img.shape[0])

        fig, axes = plt.subplots(
            B,
            3,
            figsize=(10, B * 4)
        )

        if B == 1:
            axes = axes[None]

        for i in range(B):

            x = img[i]
            att = w[i]

            # CHW→HWC
            x = x.permute(
                1,
                2,
                0
            ).numpy()

            att = att.numpy()

            # normalize
            x = (
                        x
                        - x.min()
                ) / (
                        x.max()
                        - x.min()
                        + 1e-6
                )

            att = (
                          att
                          - att.min()
                  ) / (
                          att.max()
                          - att.min()
                          + 1e-6
                  )

            # 原图
            axes[i, 0].imshow(x)
            axes[i, 0].set_title("Fundus")

            # attention
            axes[i, 1].imshow(
                att,
                cmap="jet"
            )
            axes[i, 1].set_title("w")

            # overlay
            axes[i, 2].imshow(x)

            axes[i, 2].imshow(
                att,
                cmap="jet",
                alpha=0.5
            )

            axes[i, 2].set_title("Overlay")

            for j in range(3):
                axes[i, j].axis("off")

        plt.tight_layout()

        save_path = os.path.join(
            self.fig_save_dir,
            f"overlay_{step}.png"
        )

        plt.savefig(
            save_path,
            dpi=200
        )

        plt.close()

    # -------------------------
    # train step
    # -------------------------
    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        total_acc = 0.0
        print("\n")
        for i, (img, label) in enumerate(self.train_loader):
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            pred, train_info = self.model(img, label)
            loss_stat = train_info["loss"]
            loss = loss_stat['total']

            acc = (pred.argmax(-1) == label).float().mean()
            total_loss += loss
            total_acc += acc.item()

            if i % self.print_step == 0:
                print(f"[Train] step={i} | loss={loss_stat} | acc={acc.item():.4f}")
            if i % self.vis_step == 0:
                self.visualize_attention(img.detach().cpu(), train_info['w'], i)

        return {
            "loss": total_loss / len(self.train_loader),
            "acc": total_acc / len(self.train_loader),
        }

    # -------------------------
    # eval (val/test)
    # -------------------------
    @torch.no_grad()
    def evaluate(self, is_val=True):
        dataloader = self.val_loader if is_val else self.test_loader
        if dataloader is None:
            return None

        self.model.eval()

        pred_list = []
        label_list = []

        for img, label in dataloader:
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            pred = self.model.predict(img)  # (B, C)
            pred = pred.argmax(dim=-1)

            pred_list.append(pred.cpu())
            label_list.append(label.cpu())

        pred_all = torch.cat(pred_list, dim=0)
        label_all = torch.cat(label_list, dim=0)

        metrics = compute_metrics(pred_all, label_all)

        return metrics, pred_all, label_all

    # -------------------------
    # full training
    # -------------------------
    def fit(self, epochs):
        best_qwk = -1e9

        for epoch in range(1, epochs + 1):

            # ---------------- train ----------------
            train_stat = self.train_one_epoch()

            print(
                f"[Epoch {epoch}] [Train] "
                f"loss={train_stat['loss']:.4f}, "
                f"acc={train_stat['acc']:.4f}"
            )

            # ---------------- val ----------------
            val_out = self.evaluate(is_val=True)
            if val_out is not None:
                val_metric, _, _ = val_out

                print(
                    f"[Epoch {epoch}] [Val] "
                    f"acc={val_metric['acc']:.4f}, "
                    f"f1={val_metric['f1']:.4f}, "
                    f"qwk={val_metric['qwk']:.4f}"
                )

                # save best
                if val_metric["qwk"] > best_qwk:
                    best_qwk = val_metric["qwk"]
                    ck_path = self.save("best")
                    print(f"[Save] best model -> {ck_path}")

            else:
                ck_path = self.save(epoch)
                print(f"[Save] model -> {ck_path}")

            # ---------------- test ----------------
            test_out = self.evaluate(is_val=False)
            if test_out is not None:
                test_metric, _, _ = test_out

                print(
                    f"[Epoch {epoch}] [Test] "
                    f"acc={test_metric['acc']:.4f}, "
                    f"f1={test_metric['f1']:.4f}, "
                    f"qwk={test_metric['qwk']:.4f}"
                )

    # -------------------------
    # checkpoint
    # -------------------------
    def save(self, epoch):
        path = os.path.join(self.save_dir, f"epoch_{epoch}.pth")

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.model.opt.state_dict(),
                "epoch": epoch,
            },
            path,
        )
        return path

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model"])
        self.model.opt.load_state_dict(ckpt["optimizer"])

        print(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")


if __name__ == '__main__':
    # model = TimmFeatureEncoder(model_name='convnext_small.fb_in22k_ft_in1k_384')
    model = DFViT()
    dr_image_root = "/root/autodl-tmp/baseline/AOR-DR/data/APTOS2019"
    dr_split_root = "/root/autodl-tmp/baseline/AOR-DR/data/splits"
    train_list = "APTOS_train_80.txt"
    val_list = "APTOS_val_20.txt"
    test_list = "APTOS_crossval.txt"
    img_size = 224

    dr_train_loader = make_loader(dr_image_root, splits_path=os.path.join(dr_split_root, train_list), is_train=True,
                                  img_size=img_size)
    dr_val_loader = make_loader(dr_image_root, splits_path=os.path.join(dr_split_root, val_list), is_train=False,
                                img_size=img_size)
    dr_test_loader = make_loader(dr_image_root, splits_path=os.path.join(dr_split_root, test_list), is_train=False,
                                 img_size=img_size)
    train_manager = TrainManager(model=model, train_loader=dr_train_loader, val_loader=dr_val_loader,
                                 test_loader=dr_test_loader)
    train_manager.fit(epochs=120)
