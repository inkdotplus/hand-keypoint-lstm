# -*- coding: utf-8 -*-
"""
train_lstm.py
第二阶段 Day 2 脚本：基于第一阶段 YOLO 关键点输出，训练 LSTM 做短时运动预测。

实验设计：
  - 输入窗口 T=5，预测下 1 帧（N=1）
  - 按段划分：seg1+seg2+seg3 -> 训练, seg4 -> 验证, seg5 -> 测试
  - 坐标归一化：以第 0 号点（手腕）为原点，用 bbox 对角线长度做尺度归一
  - 两组实验：
      A. baseline: 只用坐标 (42 维)
      B. velocity: 坐标 + 帧间位移 (42+42=84 维)
  - 评价指标：
      * MPJPE-2D: 反归一化后每关键点平均欧氏距离误差（像素）
      * PCK@0.05: 以 bbox 对角线 5% 为阈值的关键点正确率
  - 输出：训练日志 csv、测试集逐帧预测 csv、训练曲线 png

使用：
  python train_lstm.py --csv_dir "C:\\home\\ubuntu\\keypoints_csv" --out_dir "C:\\home\\ubuntu\\stage2_results"
"""

import argparse
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 数据加载与预处理
# ============================================================

KP_NUM = 21
COORD_DIM = KP_NUM * 2  # 42


def load_seg_csv(csv_path: Path):
    """读取一个 seg csv，返回关键点坐标 (N, 21, 2) 和 bbox (N, 4)。"""
    df = pd.read_csv(csv_path)
    N = len(df)
    kp = np.zeros((N, KP_NUM, 2), dtype=np.float32)
    for i in range(KP_NUM):
        kp[:, i, 0] = df[f"x{i}"].values
        kp[:, i, 1] = df[f"y{i}"].values
    bbox = df[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values.astype(np.float32)
    return kp, bbox


def normalize_kp(kp: np.ndarray, bbox: np.ndarray):
    """
    以第 0 号点（手腕）为原点，用 bbox 对角线长度做尺度归一化。
    kp: (N, 21, 2)
    bbox: (N, 4) [x1, y1, x2, y2]
    返回归一化后的 (N, 21, 2)，以及反归一化参数 (N, 3): (wrist_x, wrist_y, scale)
    """
    wrist = kp[:, 0:1, :].copy()  # (N, 1, 2)
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    diag = np.sqrt(w * w + h * h) + 1e-6  # (N,)
    scale = diag[:, None, None]  # (N, 1, 1)
    kp_norm = (kp - wrist) / scale
    denorm = np.concatenate([wrist.squeeze(1), diag[:, None]], axis=1)  # (N, 3)
    return kp_norm, denorm


def denormalize_kp(kp_norm: np.ndarray, denorm: np.ndarray):
    """反归一化。kp_norm: (..., 21, 2), denorm: (..., 3)。"""
    wrist_x = denorm[..., 0:1]  # (..., 1)
    wrist_y = denorm[..., 1:2]
    scale = denorm[..., 2:3]
    out = kp_norm.copy()
    out[..., 0] = kp_norm[..., 0] * scale + wrist_x
    out[..., 1] = kp_norm[..., 1] * scale + wrist_y
    return out


def build_windows(kp_norm: np.ndarray, T: int = 5, use_velocity: bool = False):
    """
    滑动窗口构造样本。
    输入 kp_norm: (N, 21, 2)
    返回:
        X: (M, T, F) F=42 或 84
        Y: (M, 42)  即下一帧所有关键点坐标（归一化后）
        idx: (M,) 每个样本对应的"目标帧"在原序列中的下标（用于反归一化）
    """
    N = kp_norm.shape[0]
    flat = kp_norm.reshape(N, -1)  # (N, 42)

    if use_velocity:
        vel = np.zeros_like(flat)
        vel[1:] = flat[1:] - flat[:-1]
        feat = np.concatenate([flat, vel], axis=1)  # (N, 84)
    else:
        feat = flat

    X, Y, idx = [], [], []
    for i in range(N - T):
        X.append(feat[i:i + T])
        Y.append(flat[i + T])  # 目标：下一帧的原始 42 维坐标
        idx.append(i + T)
    return (np.stack(X).astype(np.float32),
            np.stack(Y).astype(np.float32),
            np.array(idx, dtype=np.int64))


class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ============================================================
# 2. 模型
# ============================================================

class LSTMPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2, out_dim: int = COORD_DIM):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # x: (B, T, F)
        h, _ = self.lstm(x)
        last = h[:, -1, :]  # (B, hidden)
        return self.head(last)  # (B, 42)


# ============================================================
# 3. 评价指标
# ============================================================

def compute_mpjpe_pck(pred_norm: np.ndarray, gt_norm: np.ndarray,
                      denorm_gt: np.ndarray, pck_thresh: float = 0.05):
    """
    pred_norm, gt_norm: (M, 42) 归一化坐标
    denorm_gt: (M, 3) 目标帧的反归一化参数
    返回: MPJPE（像素）, PCK@thresh
    """
    pred_kp = pred_norm.reshape(-1, KP_NUM, 2)
    gt_kp = gt_norm.reshape(-1, KP_NUM, 2)

    pred_px = denormalize_kp(pred_kp, denorm_gt)
    gt_px = denormalize_kp(gt_kp, denorm_gt)

    err = np.linalg.norm(pred_px - gt_px, axis=-1)  # (M, 21) 像素距离
    mpjpe = err.mean()

    # PCK: 阈值 = pck_thresh * bbox 对角线
    diag = denorm_gt[:, 2:3]  # (M, 1)
    thresh = pck_thresh * diag  # (M, 1)
    correct = (err < thresh).mean()

    return float(mpjpe), float(correct)


# ============================================================
# 4. 训练流程
# ============================================================

def train_one_config(X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
                     denorm_va, denorm_te,
                     in_dim: int, epochs: int, lr: float, batch: int,
                     device: str, tag: str, out_dir: Path):
    model = LSTMPredictor(in_dim=in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    tr_loader = DataLoader(SeqDataset(X_tr, Y_tr), batch_size=batch, shuffle=True)
    va_loader = DataLoader(SeqDataset(X_va, Y_va), batch_size=batch, shuffle=False)

    log = []
    best_val_mpjpe = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            n_tr += xb.size(0)
        tr_loss /= n_tr

        # 验证
        model.eval()
        va_loss = 0.0
        n_va = 0
        va_preds = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = crit(pred, yb)
                va_loss += loss.item() * xb.size(0)
                n_va += xb.size(0)
                va_preds.append(pred.cpu().numpy())
        va_loss /= n_va
        va_preds = np.concatenate(va_preds, axis=0)
        va_mpjpe, va_pck = compute_mpjpe_pck(va_preds, Y_va, denorm_va)

        log.append({
            "epoch": ep, "train_mse": tr_loss, "val_mse": va_loss,
            "val_mpjpe_px": va_mpjpe, "val_pck@0.05": va_pck
        })
        print(f"[{tag}] ep{ep:03d}  tr_mse={tr_loss:.5f}  va_mse={va_loss:.5f}  "
              f"va_MPJPE={va_mpjpe:.2f}px  va_PCK@0.05={va_pck:.4f}")

        if va_mpjpe < best_val_mpjpe:
            best_val_mpjpe = va_mpjpe
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    # 测试（用最优验证权重）
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    te_preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), batch):
            xb = torch.from_numpy(X_te[i:i + batch]).to(device)
            pred = model(xb)
            te_preds.append(pred.cpu().numpy())
    te_preds = np.concatenate(te_preds, axis=0)
    te_mpjpe, te_pck = compute_mpjpe_pck(te_preds, Y_te, denorm_te)
    print(f"[{tag}] TEST  MPJPE={te_mpjpe:.2f}px  PCK@0.05={te_pck:.4f}")

    # 保存
    log_df = pd.DataFrame(log)
    log_df.to_csv(out_dir / f"lstm_{tag}_log.csv", index=False)

    # 测试集逐帧预测（反归一化后的像素坐标）
    te_pred_kp = denormalize_kp(te_preds.reshape(-1, KP_NUM, 2), denorm_te)
    te_gt_kp = denormalize_kp(Y_te.reshape(-1, KP_NUM, 2), denorm_te)
    rows = []
    for i in range(len(te_preds)):
        row = {"sample_idx": i}
        for k in range(KP_NUM):
            row[f"pred_x{k}"] = te_pred_kp[i, k, 0]
            row[f"pred_y{k}"] = te_pred_kp[i, k, 1]
            row[f"gt_x{k}"] = te_gt_kp[i, k, 0]
            row[f"gt_y{k}"] = te_gt_kp[i, k, 1]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"lstm_{tag}_test_predictions.csv", index=False)

    return {
        "tag": tag,
        "best_val_mpjpe": best_val_mpjpe,
        "test_mpjpe": te_mpjpe,
        "test_pck": te_pck,
        "log": log_df,
    }


# ============================================================
# 5. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--T", type=int, default=5, help="输入窗口长度")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读所有 seg
    segs = {}
    for name in ["seg1_openclose", "seg2_wave", "seg3_finger", "seg4_point", "seg5_switch"]:
        p = csv_dir / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        kp, bbox = load_seg_csv(p)
        kp_norm, denorm = normalize_kp(kp, bbox)
        segs[name] = {"kp": kp, "bbox": bbox, "kp_norm": kp_norm, "denorm": denorm}
        print(f"[INFO] {name}: {len(kp)} 帧")

    # 划分
    train_segs = ["seg1_openclose", "seg2_wave", "seg3_finger"]
    val_seg = "seg4_point"
    test_seg = "seg5_switch"

    def concat_windows(seg_list, use_vel):
        Xs, Ys, Ds = [], [], []
        for sn in seg_list:
            s = segs[sn]
            X, Y, idx = build_windows(s["kp_norm"], T=args.T, use_velocity=use_vel)
            D = s["denorm"][idx]  # 目标帧的反归一化参数
            Xs.append(X); Ys.append(Y); Ds.append(D)
        return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ds)

    # 两组实验
    results = {}
    for tag, use_vel in [("baseline", False), ("velocity", True)]:
        in_dim = COORD_DIM * (2 if use_vel else 1)
        X_tr, Y_tr, _ = concat_windows(train_segs, use_vel)
        X_va, Y_va, D_va = concat_windows([val_seg], use_vel)
        X_te, Y_te, D_te = concat_windows([test_seg], use_vel)
        print(f"\n=== {tag} ===  in_dim={in_dim}  train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")

        r = train_one_config(X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
                             D_va, D_te, in_dim=in_dim,
                             epochs=args.epochs, lr=args.lr, batch=args.batch,
                             device=device, tag=tag, out_dir=out_dir)
        results[tag] = r

    # 画训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for tag, r in results.items():
        log = r["log"]
        axes[0].plot(log["epoch"], log["train_mse"], label=f"{tag} train")
        axes[0].plot(log["epoch"], log["val_mse"], label=f"{tag} val", linestyle="--")
        axes[1].plot(log["epoch"], log["val_mpjpe_px"], label=tag)
        axes[2].plot(log["epoch"], log["val_pck@0.05"], label=tag)
    axes[0].set_title("MSE Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].set_title("Val MPJPE (px)"); axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[2].set_title("Val PCK@0.05"); axes[2].set_xlabel("epoch"); axes[2].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()

    # 汇总
    summary_rows = []
    for tag, r in results.items():
        summary_rows.append({
            "config": tag,
            "best_val_mpjpe_px": r["best_val_mpjpe"],
            "test_mpjpe_px": r["test_mpjpe"],
            "test_pck@0.05": r["test_pck"],
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    print("\n========== SUMMARY ==========")
    for row in summary_rows:
        print(row)
    print(f"\n[DONE] 全部结果已保存至 {out_dir}")


if __name__ == "__main__":
    main()
