# eval_linear_custom.py
# Linear probe on top of a frozen DINO ViT backbone,
# supports both:
#   1) online backbone (slow, more aug)
#   2) precomputed features (fast)

import os
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms as pth_transforms
import pandas as pd

import vision_transformer as vits


# ---------------------------------------------------------
# Transforms
# ---------------------------------------------------------

def make_train_transform():
    """
    Data augmentation for linear probe.
    96x96, using your dataset stats.
    """
    return pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(
            96,
            scale=(0.8, 1.0),
            interpolation=pth_transforms.InterpolationMode.BICUBIC,
        ),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(
            (0.5191, 0.4979, 0.4711),
            (0.3050, 0.3005, 0.3118),
        ),
    ])


def make_eval_transform():
    """
    Center-crop eval transform, same as knn_eval_custom.
    """
    return pth_transforms.Compose([
        pth_transforms.Resize(96, interpolation=pth_transforms.InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(96),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(
            (0.5191, 0.4979, 0.4711),
            (0.3050, 0.3005, 0.3118),
        ),
    ])


# ---------------------------------------------------------
# Dataset (CSV + image folder)
# ---------------------------------------------------------

class CSVDataset(Dataset):
    """
    root_dir/ holds images, labels_csv has:
        filename, class_id
    """
    def __init__(self, root_dir: str, labels_csv: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        df = pd.read_csv(labels_csv)
        self.filenames = df["filename"].tolist()
        self.labels = df["class_id"].tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = self.labels[idx]

        img_path = os.path.join(self.root_dir, fname)
        from PIL import Image
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label)


# ---------------------------------------------------------
# Load backbone from DINO checkpoint
# ---------------------------------------------------------

def load_backbone_from_checkpoint(ckpt_path: str, device: str = "cuda"):
    """
    Assumes ckpt structure matches your DINO training script:
      ckpt["args"].arch / patch_size
      ckpt["student"]["module.backbone.*"]
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt or "student" not in ckpt:
        raise ValueError("Checkpoint must contain 'args' and 'student' keys.")

    args = ckpt["args"]
    arch = args.arch
    patch_size = args.patch_size

    if arch not in vits.__dict__:
        raise ValueError(f"Unknown ViT arch '{arch}' in checkpoint args.")

    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)

    student_state = ckpt["student"]
    backbone_state = {}
    for k, v in student_state.items():
        if k.startswith("module.backbone."):
            new_k = k.replace("module.backbone.", "")
            backbone_state[new_k] = v

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    if len(missing) > 0:
        print("[load_backbone_from_checkpoint] Missing keys:", missing)
    if len(unexpected) > 0:
        print("[load_backbone_from_checkpoint] Unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, args


# ---------------------------------------------------------
# Linear classifier
# ---------------------------------------------------------

class LinearClassifier(nn.Module):
    """Simple linear layer on top of frozen features."""
    def __init__(self, in_dim: int, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(in_dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


# ---------------------------------------------------------
# Feature extraction from backbone
# ---------------------------------------------------------

@torch.no_grad()
def extract_backbone_features(
    model: nn.Module,
    imgs: torch.Tensor,
    arch: str,
    n_last_blocks: int,
    avgpool_patchtokens: bool,
) -> torch.Tensor:
    """
    Same idea as official eval_linear for ViT:
      - get_intermediate_layers
      - concat last n CLS tokens
      - optional avg pooled patch tokens
    """
    if "vit" in arch:
        intermediate_output = model.get_intermediate_layers(imgs, n_last_blocks)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

        if avgpool_patchtokens:
            patch_tokens = intermediate_output[-1][:, 1:, :]
            patch_avg = patch_tokens.mean(dim=1)
            output = torch.cat([output, patch_avg], dim=-1)
    else:
        output = model(imgs)

    return output


def compute_feature_dim(
    model: nn.Module,
    arch: str,
    n_last_blocks: int,
    avgpool_patchtokens: bool
) -> int:
    """
    Use ViT embed_dim formula if available, otherwise run a dummy fwd.
    """
    if "vit" in arch:
        if not hasattr(model, "embed_dim"):
            raise ValueError("ViT model missing 'embed_dim' attribute.")
        d = model.embed_dim
        dim = d * n_last_blocks
        if avgpool_patchtokens:
            dim += d
        return dim
    else:
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(2, 3, 96, 96, device=next(model.parameters()).device)
            out = model(dummy)
        return out.view(out.size(0), -1).size(1)


@torch.no_grad()
def precompute_features_for_loader(
    backbone: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    arch: str,
    n_last_blocks: int,
    avgpool_patchtokens: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run backbone once on the whole loader and cache features + labels.
    """
    backbone.eval()

    feats_list = []
    labels_list = []

    for it, (imgs, labels) in enumerate(data_loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats = extract_backbone_features(
            backbone, imgs, arch, n_last_blocks, avgpool_patchtokens
        )

        feats_list.append(feats.cpu())
        labels_list.append(labels.cpu())

        if (it + 1) % 20 == 0:
            print(f"[Precompute] Iter {it+1}/{len(data_loader)}")

    all_feats = torch.cat(feats_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    print(f"[Precompute] Done. Features shape: {all_feats.shape}")
    return all_feats, all_labels


# ---------------------------------------------------------
# Train & eval helpers
# ---------------------------------------------------------

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Small accuracy helper."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch_online(
    backbone: nn.Module,
    linear_clf: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    arch: str,
    n_last_blocks: int,
    avgpool_patchtokens: bool,
    print_freq: int = 20,
) -> Tuple[float, float]:
    """Slow path: run backbone every epoch."""
    linear_clf.train()
    backbone.eval()

    total_loss = 0.0
    total_acc1 = 0.0
    total_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    for it, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            feats = extract_backbone_features(backbone, imgs, arch, n_last_blocks, avgpool_patchtokens)

        logits = linear_clf(feats)
        loss = ce_loss(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        acc1, = accuracy(logits, labels, topk=(1,))
        total_acc1 += acc1.item() * batch_size
        total_samples += batch_size

        if (it + 1) % print_freq == 0:
            avg_loss = total_loss / total_samples
            avg_acc1 = total_acc1 / total_samples
            print(f"[Train-online] Iter {it+1}/{len(train_loader)}  "
                  f"Loss: {avg_loss:.4f}  Acc@1: {avg_acc1:.2f}%")

    epoch_loss = total_loss / total_samples
    epoch_acc1 = total_acc1 / total_samples
    return epoch_loss, epoch_acc1


@torch.no_grad()
def evaluate_online(
    backbone: nn.Module,
    linear_clf: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    arch: str,
    n_last_blocks: int,
    avgpool_patchtokens: bool,
    print_freq: int = 20,
) -> Tuple[float, float]:
    """Slow path eval: run backbone every time."""
    linear_clf.eval()
    backbone.eval()

    total_loss = 0.0
    total_acc1 = 0.0
    total_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    for it, (imgs, labels) in enumerate(val_loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats = extract_backbone_features(backbone, imgs, arch, n_last_blocks, avgpool_patchtokens)
        logits = linear_clf(feats)
        loss = ce_loss(logits, labels)

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        acc1, = accuracy(logits, labels, topk=(1,))
        total_acc1 += acc1.item() * batch_size
        total_samples += batch_size

        if (it + 1) % print_freq == 0:
            avg_loss = total_loss / total_samples
            avg_acc1 = total_acc1 / total_samples
            print(f"[Val-online]   Iter {it+1}/{len(val_loader)}  "
                  f"Loss: {avg_loss:.4f}  Acc@1: {avg_acc1:.2f}%")

    epoch_loss = total_loss / total_samples
    epoch_acc1 = total_acc1 / total_samples
    return epoch_loss, epoch_acc1


def train_one_epoch_features(
    linear_clf: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    print_freq: int = 20,
) -> Tuple[float, float]:
    """Fast path: train only on cached features."""
    linear_clf.train()

    total_loss = 0.0
    total_acc1 = 0.0
    total_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    for it, (feats, labels) in enumerate(train_loader):
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = linear_clf(feats)
        loss = ce_loss(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = feats.size(0)
        total_loss += loss.item() * batch_size
        acc1, = accuracy(logits, labels, topk=(1,))
        total_acc1 += acc1.item() * batch_size
        total_samples += batch_size

        if (it + 1) % print_freq == 0:
            avg_loss = total_loss / total_samples
            avg_acc1 = total_acc1 / total_samples
            print(f"[Train-feat] Iter {it+1}/{len(train_loader)}  "
                  f"Loss: {avg_loss:.4f}  Acc@1: {avg_acc1:.2f}%")

    epoch_loss = total_loss / total_samples
    epoch_acc1 = total_acc1 / total_samples
    return epoch_loss, epoch_acc1


@torch.no_grad()
def evaluate_features(
    linear_clf: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    print_freq: int = 20,
) -> Tuple[float, float]:
    """Fast path eval on cached features."""
    linear_clf.eval()

    total_loss = 0.0
    total_acc1 = 0.0
    total_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    for it, (feats, labels) in enumerate(val_loader):
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = linear_clf(feats)
        loss = ce_loss(logits, labels)

        batch_size = feats.size(0)
        total_loss += loss.item() * batch_size
        acc1, = accuracy(logits, labels, topk=(1,))
        total_acc1 += acc1.item() * batch_size
        total_samples += batch_size

        if (it + 1) % print_freq == 0:
            avg_loss = total_loss / total_samples
            avg_acc1 = total_acc1 / total_samples
            print(f"[Val-feat]   Iter {it+1}/{len(val_loader)}  "
                  f"Loss: {avg_loss:.4f}  Acc@1: {avg_acc1:.2f}%")

    epoch_loss = total_loss / total_samples
    epoch_acc1 = total_acc1 / total_samples
    return epoch_loss, epoch_acc1


# ---------------------------------------------------------
# One-click linear eval (two modes)
# ---------------------------------------------------------

def run_linear_eval(
    ckpt_path: str,
    eval_root: str,
    device: str = "cuda",
    n_last_blocks: int = 4,
    avgpool_patchtokens: bool = False,
    epochs: int = 50,
    batch_size: int = 1024,
    num_workers: int = 4,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    print_freq: int = 20,
    use_precomputed_features: bool = True,
):
    """
    Two modes:
      - use_precomputed_features=True: fast, cache features once
      - use_precomputed_features=False: slow, run backbone every epoch
    """
    device = torch.device(device)

    # 1) Load backbone
    backbone, args = load_backbone_from_checkpoint(ckpt_path, device=str(device))
    arch = args.arch

    # 2) Load image datasets
    train_dir = os.path.join(eval_root, "train")
    val_dir   = os.path.join(eval_root, "val")
    test_dir  = os.path.join(eval_root, "test")

    train_csv = os.path.join(eval_root, "train_labels.csv")
    val_csv   = os.path.join(eval_root, "val_labels.csv")
    test_csv  = os.path.join(eval_root, "test_labels_INTERNAL.csv")

    train_dataset_img = CSVDataset(train_dir, train_csv, transform=make_train_transform())
    val_dataset_img   = CSVDataset(val_dir,   val_csv,   transform=make_eval_transform())
    test_dataset_img  = CSVDataset(test_dir,  test_csv,  transform=make_eval_transform())

    num_labels = max(train_dataset_img.labels) + 1
    print(f"[Linear] Train:{len(train_dataset_img)}, Val:{len(val_dataset_img)}, "
          f"Test:{len(test_dataset_img)}, Classes:{num_labels}")

    if use_precomputed_features:
        # ---------------- Fast path: cache features ----------------
        train_loader_img = DataLoader(train_dataset_img, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)
        val_loader_img   = DataLoader(val_dataset_img,   batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)
        test_loader_img  = DataLoader(test_dataset_img,  batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)

        print("\n===== Precompute TRAIN features =====")
        train_feats, train_labels = precompute_features_for_loader(
            backbone, train_loader_img, device, arch, n_last_blocks, avgpool_patchtokens
        )

        print("\n===== Precompute VAL features =====")
        val_feats, val_labels = precompute_features_for_loader(
            backbone, val_loader_img, device, arch, n_last_blocks, avgpool_patchtokens
        )

        print("\n===== Precompute TEST features =====")
        test_feats, test_labels = precompute_features_for_loader(
            backbone, test_loader_img, device, arch, n_last_blocks, avgpool_patchtokens
        )

        # free backbone to save GPU memory
        del backbone
        torch.cuda.empty_cache()

        feat_dim = train_feats.size(1)
        print(f"[Linear] Feature dim: {feat_dim}")

        train_dataset_feat = TensorDataset(train_feats, train_labels)
        val_dataset_feat   = TensorDataset(val_feats,   val_labels)
        test_dataset_feat  = TensorDataset(test_feats,  test_labels)

        train_loader = DataLoader(train_dataset_feat, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_dataset_feat,   batch_size=batch_size, shuffle=False,
                                  num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_dataset_feat,  batch_size=batch_size, shuffle=False,
                                  num_workers=0, pin_memory=True)
    else:
        # ---------------- Slow path: online backbone ----------------
        train_loader = DataLoader(train_dataset_img, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_dataset_img,   batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
        test_loader  = DataLoader(test_dataset_img,  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

        feat_dim = compute_feature_dim(backbone, arch, n_last_blocks, avgpool_patchtokens)
        print(f"[Linear] Feature dim (online): {feat_dim}")

    # 3) Linear classifier
    linear_clf = LinearClassifier(feat_dim, num_labels=num_labels).to(device)

    optimizer = torch.optim.SGD(
        linear_clf.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.0
    )

    best_val_acc = 0.0

    # 4) Train epochs
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")

        if use_precomputed_features:
            train_loss, train_acc1 = train_one_epoch_features(
                linear_clf, train_loader, optimizer, device, print_freq=print_freq
            )
            val_loss, val_acc1 = evaluate_features(
                linear_clf, val_loader, device, print_freq=print_freq
            )
        else:
            train_loss, train_acc1 = train_one_epoch_online(
                backbone, linear_clf, train_loader,
                optimizer, device, arch, n_last_blocks,
                avgpool_patchtokens, print_freq=print_freq
            )
            val_loss, val_acc1 = evaluate_online(
                backbone, linear_clf, val_loader,
                device, arch, n_last_blocks,
                avgpool_patchtokens, print_freq=print_freq
            )

        scheduler.step()

        print(f"[Epoch {epoch+1}] "
              f"Train Acc: {train_acc1:.2f}% | Val Acc: {val_acc1:.2f}%")

        best_val_acc = max(best_val_acc, val_acc1)

    # 5) Final test eval
    print("\n===== Final Evaluation on TEST set =====")
    if use_precomputed_features:
        test_loss, test_acc1 = evaluate_features(
            linear_clf, test_loader, device, print_freq=print_freq
        )
    else:
        test_loss, test_acc1 = evaluate_online(
            backbone, linear_clf, test_loader,
            device, arch, n_last_blocks,
            avgpool_patchtokens, print_freq=print_freq
        )

    print(f"[Test] Test Acc@1: {test_acc1:.2f}%\n")

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc1,
    }