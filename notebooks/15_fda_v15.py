"""
V15: Offline FDA Domain Adaptation + Dev-Only Validation
========================================================
핵심 변경:
1. FDA 오프라인 전처리: train 이미지를 dev 스타일로 변환하여 디스크에 저장
2. Dev-only validation: CV를 dev 100개로만 → LB와 직접 비교 가능
3. beta 그리드 서치: 0.01, 0.03, 0.05, 0.10, 0.15, 0.20
4. 혼합 학습: FDA 변환 이미지 + 원본 train 이미지 동시 사용
"""
import copy
import gc
import math
import os
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings('ignore')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


@dataclass
class Config:
    backbone: str = 'convnextv2_base.fcmae_ft_in22k_in1k'
    exp_name: str = 'v15b_fda_5fold'
    img_size: int = 384
    epochs: int = 15
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    early_stopping_patience: int = 7
    grad_clip: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.9995
    drop_path_rate: float = 0.15
    emb_dim: int = 512
    fusion_layers: int = 2
    fusion_heads: int = 8

    # KD Loss
    kd_alpha: float = 0.7
    kd_temperature: float = 3.0

    # Multi-task
    use_multitask: bool = True
    motion_reg_weight: float = 0.20
    onset_cls_weight: float = 0.15
    severity_cls_weight: float = 0.15

    # Preprocessing
    use_center_crop: bool = True
    use_checkerboard_norm: bool = True
    use_gem: bool = True
    gem_p_init: float = 3.0

    # FDA - 핵심 변경
    fda_beta: float = 0.01  # 저주파 교체 범위 (0.01이면 색감 충분히 전환, 아티팩트 없음)
    fda_mix_ratio: float = 1.0  # train 이미지는 100% FDA 적용 (dev는 이미 target 도메인)

    n_folds: int = 5
    seed: int = 42
    tta_scales: Optional[list] = None

    data_dir: str = 'data'
    output_dir: str = 'outputs'

    def __post_init__(self):
        if self.tta_scales is None:
            self.tta_scales = [self.img_size, self.img_size + 64, self.img_size + 128]


# ============================================================
# FDA Core
# ============================================================

def fda_transfer(src_img, trg_img, beta=0.10):
    """Fourier Domain Adaptation: src의 저주파(amplitude)를 trg의 것으로 교체.

    src_img, trg_img: (H, W, 3) uint8 RGB
    beta: 교체할 저주파 영역 비율 (0~1). 클수록 더 많이 교체.
    """
    src_f = src_img.astype(np.float32)
    trg_f = trg_img.astype(np.float32)

    if src_f.shape[:2] != trg_f.shape[:2]:
        trg_f = cv2.resize(trg_f, (src_f.shape[1], src_f.shape[0]))

    result = np.zeros_like(src_f)
    for c in range(3):
        # FFT
        src_fft = np.fft.fft2(src_f[:, :, c])
        trg_fft = np.fft.fft2(trg_f[:, :, c])

        # Shift to center
        src_fft = np.fft.fftshift(src_fft)
        trg_fft = np.fft.fftshift(trg_fft)

        # Amplitude and phase
        src_amp = np.abs(src_fft)
        src_pha = np.angle(src_fft)
        trg_amp = np.abs(trg_fft)

        # Replace low-frequency amplitude
        h, w = src_f.shape[:2]
        b_h = int(h * beta)
        b_w = int(w * beta)
        cy, cx = h // 2, w // 2

        src_amp[cy - b_h:cy + b_h + 1, cx - b_w:cx + b_w + 1] = \
            trg_amp[cy - b_h:cy + b_h + 1, cx - b_w:cx + b_w + 1]

        # Reconstruct
        modified = src_amp * np.exp(1j * src_pha)
        result[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(modified)))

    return np.clip(result, 0, 255).astype(np.uint8)


def generate_fda_dataset(data_dir, beta=0.10):
    """Train 이미지를 Dev 스타일로 FDA 변환하여 저장.

    data/fda_beta010/TRAIN_0001/front.png, top.png 형식으로 저장.
    """
    data_dir = Path(data_dir)
    beta_str = f'{beta:.2f}'.replace('.', '')
    out_dir = data_dir / f'fda_beta{beta_str}'

    if out_dir.exists() and len(list(out_dir.iterdir())) >= 990:
        print(f'FDA dataset already exists at {out_dir} ({len(list(out_dir.iterdir()))} folders)')
        return out_dir

    # Load all dev images as references
    dev_dir = data_dir / 'dev'
    dev_imgs = {}
    for sid_dir in sorted(dev_dir.iterdir()):
        if not sid_dir.is_dir():
            continue
        front = cv2.cvtColor(cv2.imread(str(sid_dir / 'front.png')), cv2.COLOR_BGR2RGB)
        top = cv2.cvtColor(cv2.imread(str(sid_dir / 'top.png')), cv2.COLOR_BGR2RGB)
        dev_imgs[sid_dir.name] = {'front': front, 'top': top}

    dev_keys = list(dev_imgs.keys())
    print(f'Loaded {len(dev_keys)} dev reference images')

    # Transform each train image
    train_dir = data_dir / 'train'
    out_dir.mkdir(parents=True, exist_ok=True)

    for sid_dir in tqdm(sorted(train_dir.iterdir()), desc=f'FDA beta={beta}'):
        if not sid_dir.is_dir():
            continue

        out_sample = out_dir / sid_dir.name
        out_sample.mkdir(exist_ok=True)

        # Random dev reference for this sample
        ref_key = random.choice(dev_keys)
        ref = dev_imgs[ref_key]

        for view in ['front', 'top']:
            src = cv2.cvtColor(cv2.imread(str(sid_dir / f'{view}.png')), cv2.COLOR_BGR2RGB)
            transferred = fda_transfer(src, ref[view], beta=beta)
            cv2.imwrite(str(out_sample / f'{view}.png'),
                        cv2.cvtColor(transferred, cv2.COLOR_RGB2BGR))

    print(f'FDA dataset saved to {out_dir}')
    return out_dir


# ============================================================
# Preprocessing (v14 그대로)
# ============================================================

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
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    fg_mask = ((sat > 30) | (val < 80) | (val > 220)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    bg_mask = cv2.bitwise_not(fg_mask)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.bitwise_and(edges, bg_mask)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=24, maxLineGap=6)
    if lines is None or len(lines) < 10:
        return None
    angles = []
    for line in lines[:400]:
        x1, y1, x2, y2 = line[0]
        angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 90)
    hist, bins = np.histogram(angles, bins=90, range=(0, 90))
    peak_angle = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2
    if hist.max() / (hist.sum() + 1e-6) < 0.08:
        return None
    if peak_angle > 45:
        peak_angle -= 90
    return peak_angle


def normalize_top_rotation(img):
    angle = estimate_checkerboard_rotation(img)
    if angle is None:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(128, 128, 128))


def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.2), contrast_limit=(-0.35, 0.35), p=0.8),
        A.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.20, hue=0.04, p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.35),
        A.Perspective(scale=(0.02, 0.10), p=0.35),
        A.Affine(scale=(0.92, 1.08), translate_percent=(-0.05, 0.05), rotate=(-7, 7), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(num_holes_range=(1, 3),
                        hole_height_range=(int(img_size * 0.02), int(img_size * 0.08)),
                        hole_width_range=(int(img_size * 0.02), int(img_size * 0.08)),
                        fill=0, p=0.10),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'top': 'image'}, is_check_shapes=False)


def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'top': 'image'}, is_check_shapes=False)


# ============================================================
# Dataset - FDA 버전
# ============================================================

class FDADataset(Dataset):
    """Train: FDA 변환 이미지 + 원본을 mix_ratio로 혼합.
    Val/Test: 원본 그대로."""

    def __init__(self, df, data_dir, transforms=None, is_test=False, cfg=None,
                 fda_dir=None, use_fda=False):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.is_test = is_test
        self.cfg = cfg or Config()
        self.fda_dir = Path(fda_dir) if fda_dir else None
        self.use_fda = use_fda and fda_dir is not None

    def __len__(self):
        return len(self.df)

    def _load_images(self, sample_id, split):
        """이미지 로드. FDA 모드일 때 train 이미지는 확률적으로 FDA 버전 사용."""
        if self.is_test or split == 'test':
            base = self.data_dir / 'test' / sample_id
        elif split == 'dev':
            base = self.data_dir / 'dev' / sample_id
        else:
            # Train: FDA 혼합
            if self.use_fda and random.random() < self.cfg.fda_mix_ratio:
                base = self.fda_dir / sample_id
                if not base.exists():
                    base = self.data_dir / 'train' / sample_id
            else:
                base = self.data_dir / 'train' / sample_id

        front = cv2.cvtColor(cv2.imread(str(base / 'front.png')), cv2.COLOR_BGR2RGB)
        top = cv2.cvtColor(cv2.imread(str(base / 'top.png')), cv2.COLOR_BGR2RGB)
        return front, top

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['id']
        split = row.get('split', 'train')

        front, top = self._load_images(sample_id, split)

        if self.cfg.use_center_crop:
            front = center_physics_crop(front, 'front')
            top = center_physics_crop(top, 'top')
        if self.cfg.use_checkerboard_norm:
            top = normalize_top_rotation(top)

        if self.transforms:
            augmented = self.transforms(image=front, top=top)
            front = augmented['image']
            top = augmented['top']

        result = {'front': front, 'top': top, 'id': sample_id}
        if not self.is_test:
            result['label'] = int(row['label_int'])
            result['soft_target'] = float(row.get('soft_target', row['label_int']))
            result['max_diff_first'] = float(row['max_diff_first']) if pd.notna(row.get('max_diff_first')) else -1.0
            result['mean_diff_prev'] = float(row['mean_diff_prev']) if pd.notna(row.get('mean_diff_prev')) else -1.0
            result['onset_bucket'] = int(row['onset_bucket']) if pd.notna(row.get('onset_bucket')) else -1
            result['severity_bucket'] = int(row['severity_bucket']) if pd.notna(row.get('severity_bucket')) else -1
        return result


# ============================================================
# Model (v14 그대로)
# ============================================================

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (x.size(-2), x.size(-1))).pow(1.0 / self.p).flatten(1)


class DualStreamModelV3(nn.Module):
    def __init__(self, backbone_name, emb_dim=512, drop_path_rate=0.15,
                 use_gem=True, gem_p=3.0, fusion_layers=2, fusion_heads=8):
        super().__init__()
        self.emb_dim = emb_dim
        self.backbone_front = timm.create_model(backbone_name, pretrained=True,
                                                 num_classes=0, drop_path_rate=drop_path_rate)
        self.backbone_top = timm.create_model(backbone_name, pretrained=True,
                                               num_classes=0, drop_path_rate=drop_path_rate)
        for bb in [self.backbone_front, self.backbone_top]:
            if hasattr(bb, 'set_grad_checkpointing'):
                bb.set_grad_checkpointing(True)
        feat_dim = self.backbone_front.num_features

        self.gem_front = GeM(p=gem_p) if use_gem else nn.AdaptiveAvgPool2d(1)
        self.gem_top = GeM(p=gem_p) if use_gem else nn.AdaptiveAvgPool2d(1)
        self.proj_front = nn.Sequential(nn.Linear(feat_dim, emb_dim), nn.GELU(), nn.Dropout(0.15))
        self.proj_top = nn.Sequential(nn.Linear(feat_dim, emb_dim), nn.GELU(), nn.Dropout(0.15))
        self.view_embed = nn.Parameter(torch.randn(2, emb_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=fusion_heads, dim_feedforward=emb_dim * 4,
            dropout=0.10, batch_first=True, activation='gelu', norm_first=True)
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=fusion_layers)
        self.norm = nn.LayerNorm(emb_dim)

        fused_dim = emb_dim * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, emb_dim),
            nn.GELU(), nn.Dropout(0.20), nn.Linear(emb_dim, 1))
        self.motion_head = nn.Sequential(
            nn.Linear(fused_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 2))
        self.onset_head = nn.Sequential(
            nn.Linear(fused_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 4))
        self.severity_head = nn.Sequential(
            nn.Linear(fused_dim, emb_dim // 2), nn.GELU(), nn.Linear(emb_dim // 2, 4))

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
        return {
            'logit': self.classifier(feat).squeeze(1),
            'motion_reg': self.motion_head(feat),
            'onset_logit': self.onset_head(feat),
            'severity_logit': self.severity_head(feat),
        }


# ============================================================
# Loss / EMA / Scheduler (v14 그대로)
# ============================================================

class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.9995):
        super().__init__()
        self.module = copy.deepcopy(model).cpu()
        self.module.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data.cpu(), alpha=1.0 - self.decay)


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.current_step = 0
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / max(self.warmup_steps, 1)
        else:
            progress = (self.current_step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale


def compute_loss_kd(outputs, batch, cfg):
    logit = outputs['logit']
    hard_label = batch['label'].float().to(logit.device)
    teacher_soft = batch['soft_target'].float().to(logit.device)

    kd_loss = F.binary_cross_entropy_with_logits(logit, teacher_soft)
    hard_loss = F.binary_cross_entropy_with_logits(logit, hard_label)
    main_loss = cfg.kd_alpha * kd_loss + (1 - cfg.kd_alpha) * hard_loss
    total_loss = main_loss

    if cfg.use_multitask:
        max_df = batch['max_diff_first'].float().to(logit.device)
        mean_dp = batch['mean_diff_prev'].float().to(logit.device)
        valid_motion = max_df >= 0
        if valid_motion.any():
            motion_tgt = torch.stack([max_df[valid_motion] / 10.0,
                                       mean_dp[valid_motion] / 0.15], dim=1).clamp(0, 2)
            total_loss = total_loss + cfg.motion_reg_weight * F.smooth_l1_loss(
                outputs['motion_reg'][valid_motion], motion_tgt)
        onset = batch['onset_bucket'].long().to(logit.device)
        valid_onset = onset >= 0
        if valid_onset.any():
            total_loss = total_loss + cfg.onset_cls_weight * F.cross_entropy(
                outputs['onset_logit'][valid_onset], onset[valid_onset])
        sev = batch['severity_bucket'].long().to(logit.device)
        valid_sev = sev >= 0
        if valid_sev.any():
            total_loss = total_loss + cfg.severity_cls_weight * F.cross_entropy(
                outputs['severity_logit'][valid_sev], sev[valid_sev])

    return total_loss


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def fit(self, logits, y_true, max_iter=200):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(dev)
        x = torch.tensor(logits, dtype=torch.float32, device=dev)
        y = torch.tensor(y_true, dtype=torch.float32, device=dev)
        opt = torch.optim.LBFGS(self.parameters(), lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(x / self.temperature, y)
            loss.backward()
            return loss

        opt.step(closure)
        return max(float(self.temperature.detach().cpu().item()), 0.01)


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg, ema_model=None):
    model.train()
    total_loss = 0
    for batch in loader:
        front = batch['front'].cuda(non_blocking=True)
        top = batch['top'].cuda(non_blocking=True)

        with autocast('cuda'):
            outputs = model(front, top)
            loss = compute_loss_kd(outputs, batch, cfg)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if ema_model is not None:
            ema_model.update(model)

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in loader:
        front = batch['front'].cuda(non_blocking=True)
        top = batch['top'].cuda(non_blocking=True)

        with autocast('cuda'):
            outputs = model(front, top)

        all_logits.append(outputs['logit'].cpu().numpy())
        all_labels.append(batch['label'].numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = sigmoid_np(logits)
    logloss = log_loss(labels, probs, labels=[0, 1])
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    return logloss, auc, logits


def train_and_evaluate(cfg, train_df, val_df, fda_dir, fold_name=''):
    """Train on train_df (FDA), validate on val_df (dev)."""
    print(f'\n{"="*60}')
    print(f'Training: {len(train_df)} samples, Validation: {len(val_df)} samples')
    print(f'FDA dir: {fda_dir}, beta={cfg.fda_beta}, mix_ratio={cfg.fda_mix_ratio}')
    print(f'{"="*60}')

    train_ds = FDADataset(train_df, cfg.data_dir,
                          transforms=get_train_transforms(cfg.img_size),
                          cfg=cfg, fda_dir=fda_dir, use_fda=True)
    val_ds = FDADataset(val_df, cfg.data_dir,
                        transforms=get_val_transforms(cfg.img_size),
                        cfg=cfg, use_fda=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = DualStreamModelV3(
        cfg.backbone, emb_dim=cfg.emb_dim, drop_path_rate=cfg.drop_path_rate,
        use_gem=cfg.use_gem, gem_p=cfg.gem_p_init,
        fusion_layers=cfg.fusion_layers, fusion_heads=cfg.fusion_heads
    ).cuda()

    ema_model = ModelEmaV2(model, cfg.ema_decay) if cfg.use_ema else None

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': cfg.lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': cfg.lr},
    ], weight_decay=cfg.weight_decay)

    scaler = GradScaler('cuda')
    scheduler = CosineWarmupScheduler(optimizer, cfg.warmup_epochs, cfg.epochs, len(train_loader))

    best_loss = float('inf')
    best_logits = None
    patience_counter = 0
    exp_dir = Path(cfg.output_dir) / cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg, ema_model)

        # Validate with EMA model if available
        eval_model = ema_model.module.cuda() if ema_model else model
        val_loss, val_auc, val_logits = validate(eval_model, val_loader)
        if ema_model:
            ema_model.module.cpu()

        improved = '***' if val_loss < best_loss else ''
        print(f'  Epoch {epoch:2d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_auc={val_auc:.4f} {improved}')

        if val_loss < best_loss:
            best_loss = val_loss
            best_logits = val_logits.copy()
            patience_counter = 0
            # Save best model
            save_model = ema_model.module if ema_model else model
            torch.save(save_model.state_dict(), exp_dir / f'best_model{fold_name}.pt')
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f'  Early stopping at epoch {epoch}')
                break

    # Cleanup
    del model, optimizer, scaler
    if ema_model:
        del ema_model
    gc.collect()
    torch.cuda.empty_cache()

    return best_loss, best_logits


@torch.no_grad()
def predict_test(cfg, model_path):
    """Test inference with TTA."""
    test_df = pd.read_csv(Path(cfg.data_dir) / 'sample_submission.csv')

    model = DualStreamModelV3(
        cfg.backbone, emb_dim=cfg.emb_dim, drop_path_rate=0,
        use_gem=cfg.use_gem, gem_p=cfg.gem_p_init,
        fusion_layers=cfg.fusion_layers, fusion_heads=cfg.fusion_heads
    ).cuda()
    model.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=True))
    model.eval()

    all_preds = []
    for scale in cfg.tta_scales:
        ds = FDADataset(test_df, cfg.data_dir,
                        transforms=get_val_transforms(scale),
                        is_test=True, cfg=cfg)
        loader = DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=True)
        logits = []
        for batch in loader:
            front = batch['front'].cuda(non_blocking=True)
            top = batch['top'].cuda(non_blocking=True)
            with autocast('cuda'):
                out = model(front, top)
            logits.append(out['logit'].cpu().numpy())

        # HFlip TTA
        ds_flip = FDADataset(test_df, cfg.data_dir,
                             transforms=A.Compose([
                                 A.Resize(scale, scale),
                                 A.HorizontalFlip(p=1.0),
                                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                 ToTensorV2(),
                             ], additional_targets={'top': 'image'}, is_check_shapes=False),
                             is_test=True, cfg=cfg)
        loader_flip = DataLoader(ds_flip, batch_size=cfg.batch_size * 2, shuffle=False,
                                 num_workers=0, pin_memory=True)
        logits_flip = []
        for batch in loader_flip:
            front = batch['front'].cuda(non_blocking=True)
            top = batch['top'].cuda(non_blocking=True)
            with autocast('cuda'):
                out = model(front, top)
            logits_flip.append(out['logit'].cpu().numpy())

        all_preds.append(np.concatenate(logits))
        all_preds.append(np.concatenate(logits_flip))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    avg_logits = np.mean(all_preds, axis=0)
    return avg_logits, test_df


# ============================================================
# Main
# ============================================================

def main():
    cfg = Config()
    seed_everything(cfg.seed)
    device = torch.device('cuda')

    print(f'Config: {cfg.exp_name}')
    print(f'Backbone: {cfg.backbone}')
    print(f'FDA beta: {cfg.fda_beta}, mix_ratio: {cfg.fda_mix_ratio}')

    # Step 1: Generate FDA dataset (train only - dev는 이미 target 도메인)
    print('\n' + '=' * 60)
    print('STEP 1: Generate FDA dataset')
    print('=' * 60)
    fda_dir = generate_fda_dataset(cfg.data_dir, beta=cfg.fda_beta)

    # Step 2: Load data - Train + Dev 합치기 (v14처럼)
    data_dir = Path(cfg.data_dir)
    train_df = pd.read_csv(data_dir / 'train.csv')
    dev_df = pd.read_csv(data_dir / 'dev.csv')

    train_df['split'] = 'train'
    dev_df['split'] = 'dev'

    all_df = pd.concat([train_df, dev_df], ignore_index=True)
    all_df['label_int'] = (all_df['label'] == 'unstable').astype(int)

    # Teacher soft labels (train만 있음, dev는 hard label 사용)
    teacher_csv = data_dir / 'video_teacher_soft_labels.csv'
    if teacher_csv.exists():
        teacher_df = pd.read_csv(teacher_csv)
        all_df = all_df.merge(teacher_df[['id', 'teacher_soft_target']], on='id', how='left')
        all_df['soft_target'] = all_df['teacher_soft_target'].fillna(all_df['label_int'].astype(float))
    else:
        all_df['soft_target'] = all_df['label_int'].astype(float)

    # Motion targets
    motion_csv = data_dir / 'motion_targets.csv'
    if motion_csv.exists():
        motion_df = pd.read_csv(motion_csv)
        all_df = all_df.merge(motion_df, on='id', how='left')

    for col in ['max_diff_first', 'mean_diff_prev', 'onset_bucket', 'severity_bucket']:
        if col not in all_df.columns:
            all_df[col] = np.nan

    print(f'Total: {len(all_df)} (train: {len(train_df)}, dev: {len(dev_df)})')
    print(f'Unstable ratio: {all_df["label_int"].mean():.3f}')

    # Step 3: 5-Fold CV (Train+Dev 전부 사용)
    print('\n' + '=' * 60)
    print('STEP 3: 5-Fold CV with FDA')
    print('=' * 60)

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    exp_dir = Path(cfg.output_dir) / cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    oof_logits = np.zeros(len(all_df))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_df, all_df['label_int'])):
        print(f'\n--- FOLD {fold} ---')
        fold_train = all_df.iloc[train_idx].reset_index(drop=True)
        fold_val = all_df.iloc[val_idx].reset_index(drop=True)

        best_loss, best_logits = train_and_evaluate(
            cfg, fold_train, fold_val, fda_dir, fold_name=f'_fold{fold}')

        oof_logits[val_idx] = best_logits
        fold_scores.append(best_loss)
        print(f'Fold {fold}: LogLoss={best_loss:.4f}')

    # OOF 결과
    oof_probs = sigmoid_np(oof_logits)
    overall_loss = log_loss(all_df['label_int'].values, oof_probs, labels=[0, 1])
    overall_auc = roc_auc_score(all_df['label_int'].values, oof_probs)
    print(f'\n{"="*60}')
    print('OVERALL CV RESULTS')
    for i, s in enumerate(fold_scores):
        print(f'  Fold {i}: LogLoss = {s:.4f}')
    print(f'  Mean:   LogLoss = {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}')
    print(f'  Overall LogLoss = {overall_loss:.4f}')
    print(f'  Overall AUC     = {overall_auc:.4f}')

    # Step 4: Temperature scaling on OOF
    print('\n' + '=' * 60)
    print('STEP 4: Temperature Scaling')
    print('=' * 60)
    ts = TemperatureScaler()
    best_temp = ts.fit(oof_logits, all_df['label_int'].values)
    cal_probs = sigmoid_np(oof_logits / best_temp)
    cal_loss = log_loss(all_df['label_int'].values, cal_probs, labels=[0, 1])
    print(f'Temperature: {best_temp:.4f}')
    print(f'Calibrated OOF LogLoss: {cal_loss:.4f}')

    # Step 5: Test inference (5-fold 평균)
    print('\n' + '=' * 60)
    print('STEP 5: Test Inference (5-fold ensemble)')
    print('=' * 60)
    all_test_logits = []
    for fold in range(cfg.n_folds):
        model_path = exp_dir / f'best_model_fold{fold}.pt'
        fold_logits, test_df = predict_test(cfg, model_path)
        all_test_logits.append(fold_logits)
        print(f'Fold {fold}: mean logit = {fold_logits.mean():.4f}')

    avg_test_logits = np.mean(all_test_logits, axis=0)
    test_probs = sigmoid_np(avg_test_logits / best_temp)
    test_probs = np.clip(test_probs, 1e-7, 1 - 1e-7)

    # Save submission
    sub_dir = Path('submissions')
    sub_dir.mkdir(exist_ok=True)
    sub = test_df[['id']].copy()
    sub['unstable_prob'] = test_probs
    sub['stable_prob'] = 1 - test_probs
    sub_path = sub_dir / f'{cfg.exp_name}_submission.csv'
    sub.to_csv(sub_path, index=False)
    print(f'\nSubmission saved: {sub_path}')
    print(f'Test unstable mean: {test_probs.mean():.4f}')
    print(f'CV LogLoss: {overall_loss:.4f}')
    print(f'Calibrated CV LogLoss: {cal_loss:.4f}')


if __name__ == '__main__':
    main()
