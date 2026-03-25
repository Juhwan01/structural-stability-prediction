"""v14 inference only - standalone script"""
import gc, json, copy, math, os, random, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
gc.collect()

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

@dataclass
class Config:
    backbone: str = 'convnextv2_base.fcmae_ft_in22k_in1k'
    exp_name: str = 'v14_balanced_fold'
    img_size: int = 384
    batch_size: int = 4
    emb_dim: int = 512
    fusion_layers: int = 2
    fusion_heads: int = 8
    use_gem: bool = True
    gem_p_init: float = 3.0
    drop_path_rate: float = 0.15
    use_center_crop: bool = True
    use_checkerboard_norm: bool = True
    n_folds: int = 5
    tta_scales: Optional[list] = None
    data_dir: str = 'data'
    output_dir: str = 'outputs'
    def __post_init__(self):
        if self.tta_scales is None:
            self.tta_scales = [self.img_size, self.img_size + 64, self.img_size + 128]

cfg = Config()
device = torch.device('cuda')
exp_dir = Path(cfg.output_dir) / cfg.exp_name

# --- Preprocessing ---
def center_physics_crop(img, view):
    h, w = img.shape[:2]
    if view == 'front':
        x1, y1 = int(0.25 * w), int(0.20 * h)
        x2, y2 = int(0.75 * w), int(0.88 * h)
    else:
        x1, y1 = int(0.29 * w), int(0.29 * h)
        x2, y2 = int(0.71 * w), int(0.71 * h)
    return img[y1:y2, x1:x2]

def estimate_checkerboard_rotation(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    fg_mask = ((sat > 30) | (val < 80) | (val > 220)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    bg_mask = cv2.bitwise_not(fg_mask)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.bitwise_and(edges, bg_mask)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=24, maxLineGap=6)
    if lines is None or len(lines) < 10:
        return None
    angles = [np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) % 90 for l in lines[:400]]
    hist, bins = np.histogram(angles, bins=90, range=(0, 90))
    peak = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2
    if hist.max() / (hist.sum() + 1e-6) < 0.08:
        return None
    return peak - 90 if peak > 45 else peak

def normalize_top_rotation(img):
    angle = estimate_checkerboard_rotation(img)
    if angle is None:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(128, 128, 128))

def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'top': 'image'}, is_check_shapes=False)

# --- Dataset ---
class StructuralDatasetV3(Dataset):
    def __init__(self, df, data_dir, transforms=None, cfg=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.cfg = cfg
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row['id']
        base = self.data_dir / 'test' / sid
        front = cv2.cvtColor(cv2.imread(str(base / 'front.png')), cv2.COLOR_BGR2RGB)
        top = cv2.cvtColor(cv2.imread(str(base / 'top.png')), cv2.COLOR_BGR2RGB)
        if self.cfg.use_center_crop:
            front = center_physics_crop(front, 'front')
            top = center_physics_crop(top, 'top')
        if self.cfg.use_checkerboard_norm:
            top = normalize_top_rotation(top)
        if self.transforms:
            aug = self.transforms(image=front, top=top)
            front, top = aug['image'], aug['top']
        return {'front': front, 'top': top, 'id': sid}

# --- Model ---
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).flatten(1)

class DualStreamModelV3(nn.Module):
    def __init__(self, backbone_name, emb_dim=512, drop_path_rate=0.15,
                 use_gem=True, gem_p=3.0, fusion_layers=2, fusion_heads=8):
        super().__init__()
        self.emb_dim = emb_dim
        self.backbone_front = timm.create_model(backbone_name, pretrained=False, num_classes=0, drop_path_rate=drop_path_rate)
        self.backbone_top = timm.create_model(backbone_name, pretrained=False, num_classes=0, drop_path_rate=drop_path_rate)
        feat_dim = self.backbone_front.num_features
        self.gem_front = GeM(p=gem_p) if use_gem else nn.AdaptiveAvgPool2d(1)
        self.gem_top = GeM(p=gem_p) if use_gem else nn.AdaptiveAvgPool2d(1)
        self.proj_front = nn.Sequential(nn.Linear(feat_dim, emb_dim), nn.GELU(), nn.Dropout(0.15))
        self.proj_top = nn.Sequential(nn.Linear(feat_dim, emb_dim), nn.GELU(), nn.Dropout(0.15))
        self.view_embed = nn.Parameter(torch.randn(2, emb_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=fusion_heads, dim_feedforward=emb_dim * 4,
            dropout=0.10, batch_first=True, activation='gelu', norm_first=True)
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=fusion_layers)
        self.norm = nn.LayerNorm(emb_dim)
        fused_dim = emb_dim * 3
        self.classifier = nn.Sequential(nn.LayerNorm(fused_dim), nn.Linear(fused_dim, emb_dim), nn.GELU(), nn.Dropout(0.20), nn.Linear(emb_dim, 1))
        self.motion_head = nn.Sequential(nn.Linear(fused_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 2))
        self.onset_head = nn.Sequential(nn.Linear(fused_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 4))
        self.severity_head = nn.Sequential(nn.Linear(fused_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 4))
    def forward(self, front, top):
        f_feat = self.backbone_front(front)
        t_feat = self.backbone_top(top)
        if f_feat.ndim > 2:
            f_feat = self.gem_front(f_feat).flatten(1)
            t_feat = self.gem_top(t_feat).flatten(1)
        f_emb = self.proj_front(f_feat)
        t_emb = self.proj_top(t_feat)
        tokens = torch.stack([f_emb + self.view_embed[0], t_emb + self.view_embed[1]], dim=1)
        fused = self.fusion(tokens)
        fused_mean = self.norm(fused.mean(dim=1))
        feat = torch.cat([f_emb, t_emb, fused_mean], dim=1)
        return {'logit': self.classifier(feat).squeeze(1)}

# --- Run Inference ---
print("Starting v14 inference...")
test_df = pd.read_csv(Path(cfg.data_dir) / 'sample_submission.csv')
test_df['split'] = 'test'
all_fold_preds = []

for fold in range(cfg.n_folds):
    fold_dir = exp_dir / f'fold{fold}'
    print(f'Fold {fold} inference...')
    torch.cuda.empty_cache()
    gc.collect()

    model = DualStreamModelV3(cfg.backbone, emb_dim=cfg.emb_dim, drop_path_rate=0,
        use_gem=cfg.use_gem, gem_p=cfg.gem_p_init,
        fusion_layers=cfg.fusion_layers, fusion_heads=cfg.fusion_heads).to(device)
    model.load_state_dict(torch.load(fold_dir / 'best_model.pt', weights_only=True))
    model.eval()

    with open(fold_dir / 'temperature.json') as f:
        temp = json.load(f)['temperature']

    fold_tta_preds = []
    for scale in cfg.tta_scales:
        for flip in [False, True]:
            transforms = get_val_transforms(scale)
            test_ds = StructuralDatasetV3(test_df, Path(cfg.data_dir), transforms, cfg=cfg)
            test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)
            tta_logits = []
            with torch.no_grad():
                for batch in test_loader:
                    front = batch['front'].to(device)
                    top = batch['top'].to(device)
                    if flip:
                        front = torch.flip(front, dims=[3])
                        top = torch.flip(top, dims=[3])
                    with autocast('cuda', dtype=torch.bfloat16):
                        out = model(front, top)
                    tta_logits.append(out['logit'].float().cpu())
            tta_logits = torch.cat(tta_logits).numpy()
            fold_tta_preds.append(sigmoid_np(tta_logits / temp))

    fold_mean = np.mean(fold_tta_preds, axis=0)
    all_fold_preds.append(fold_mean)
    print(f'  Fold {fold}: mean pred = {fold_mean.mean():.4f}')
    del model
    torch.cuda.empty_cache()

test_preds = np.mean(all_fold_preds, axis=0)
np.save(exp_dir / 'test_preds.npy', test_preds)

# Submission
unstable_prob = np.clip(test_preds, 1e-6, 1 - 1e-6)
test_df['unstable_prob'] = unstable_prob
test_df['stable_prob'] = 1.0 - unstable_prob
sub_path = Path('submissions') / f'{cfg.exp_name}_submission.csv'
sub_path.parent.mkdir(parents=True, exist_ok=True)
test_df[['id', 'unstable_prob', 'stable_prob']].to_csv(sub_path, index=False)
print(f'\nSubmission saved: {sub_path}')
print(f'Mean unstable prob: {unstable_prob.mean():.4f}')
