
# knn_eval_custom.py
# Modified from official DINO knn_eval.py

import os
import sys
import torch
from torch import nn
from torchvision import transforms as pth_transforms
from torch.utils.data import DataLoader, Dataset
# from torchvision.datasets import ImageFolder

import pandas as pd

import vision_transformer as vits
    

# ---------------------------------------------------------
# Build eval transform (same as official)
# ---------------------------------------------------------
def make_eval_transform():
    return pth_transforms.Compose([
        pth_transforms.Resize(96, interpolation=pth_transforms.InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(96),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


# ---------------------------------------------------------
# Load backbone from checkpoint
# ---------------------------------------------------------
def load_backbone_from_checkpoint(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)

    student_state = ckpt["student"]
    backbone_state = {}
    for k, v in student_state.items():
        if k.startswith("module.backbone."):
            new_k = k.replace("module.backbone.", "")
            backbone_state[new_k] = v

    model.load_state_dict(backbone_state, strict=False)
    model.to(device)
    model.eval()

    return model, args

# ---------------------------------------------------------
# Define dataset
# ---------------------------------------------------------

class CSVDataset(Dataset):
    def __init__(self, root_dir, labels_csv, transform=None):

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

        return img, label
        
# ---------------------------------------------------------
# Extract features (single split)
# ---------------------------------------------------------
@torch.no_grad()
def extract_features(model, img_dir, labels_csv, batch_size=256, num_workers=2, device="cuda"):
    transform = make_eval_transform()

    dataset = CSVDataset(img_dir, labels_csv, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    features_list = []
    labels_list = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = model(imgs)
        features_list.append(feats.cpu())
        labels_list.extend(labels.tolist())

    features = torch.cat(features_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long)

    return features, labels


# ---------------------------------------------------------
# Weighted kNN classifier (official logic)
# ---------------------------------------------------------
@torch.no_grad()
def knn_classifier(train_feats, train_labels, test_feats, test_labels, k=20, T=0.07):
    train_feats = nn.functional.normalize(train_feats, dim=1)
    test_feats  = nn.functional.normalize(test_feats, dim=1)

    train_feats = train_feats.t()   # [D, N_train]
    device = train_feats.device

    num_classes = train_labels.max().item() + 1
    num_test = test_labels.size(0)
    retrieval_onehot = torch.zeros(k, num_classes).to(device)

    top1 = top5 = total = 0

    for i in range(0, num_test):
        feat = test_feats[i:i+1]  # [1, D]

        # similarity
        sim = torch.mm(feat, train_feats)  # [1, N_train]
        dist, idx = sim.topk(k, largest=True, sorted=True)

        neighbors = train_labels[idx[0]]

        retrieval_onehot.zero_()
        retrieval_onehot.scatter_(1, neighbors.view(-1, 1), 1)

        probs = (retrieval_onehot * (dist[0].div(T).exp().view(-1, 1))).sum(0)
        pred = probs.argmax().item()

        total += 1
        if pred == test_labels[i].item():
            top1 += 1

    top1 = top1 * 100.0 / total
    return top1


# ---------------------------------------------------------
# One-click kNN eval
# ---------------------------------------------------------
def run_knn_eval(ckpt_path, eval_root, device="cuda", k=20):
    """
    eval_root/
       train/
       val/
       train_labels.csv
       val_labels.csv
    """

    model, args = load_backbone_from_checkpoint(ckpt_path, device)

    train_dir = os.path.join(eval_root, "train")
    val_dir   = os.path.join(eval_root, "val")

    train_csv = os.path.join(eval_root, "train_labels.csv")
    val_csv   = os.path.join(eval_root, "val_labels.csv")

    print("[kNN] Extracting train features...")
    train_feats, train_labels = extract_features(model, train_dir, train_csv, device=device)
    print("[kNN] Extracting val features...")
    val_feats, val_labels     = extract_features(model, val_dir,   val_csv,   device=device)

    # move to GPU
    train_feats = train_feats.to(device)
    val_feats   = val_feats.to(device)
    train_labels = train_labels.to(device)
    val_labels   = val_labels.to(device)

    print("[kNN] Running weighted kNN...")
    top1 = knn_classifier(train_feats, train_labels, val_feats, val_labels, k=k, T=0.07)

    print(f"[kNN] {k}-NN Top1 Acc: {top1:.2f}%")
    return top1
