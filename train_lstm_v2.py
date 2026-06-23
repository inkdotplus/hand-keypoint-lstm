# -*- coding: utf-8 -*-
"""
train_lstm_v2.py
第二阶段 Day 2 脚本（修订版）。

相比第一版的改动：
  1. 数据划分：每段内部按 70%/15%/15% 切（训练/验证/测试），5 段的对应份拼在一起。
     相比原来的"跨段划分"，这样训练与测试动作类型一致，更能反映模型学习运动规律的能力。
  2. 增加朴素基线（naive copy-previous）作为对照，与 LSTM baseline、LSTM velocity 一并汇报。
  3. 同时保留跨段划分的实验结果，文件名后缀 _crossseg；段内划分结果后缀 _within。
     论文里两种结果都报告，形成完整的对比分析。

使用：
  python train_lstm_v2.py --csv_dir "C:\\home\\ubuntu\\keypoints_csv" --out_dir "C:\\home\\ubuntu\\stage2_results_v2"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


KP_NUM = 21
COORD_DIM = KP_NUM * 2


# ============================================================
# 数据加载与预处理（与第一版一致）
# ============================================================

def load_seg_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    N = len(df)
    kp = np.zeros((N, KP_NUM, 2), dtype=np.float32)
    for i in range(KP_NUM):
        kp[:, i, 0] = df[f"x{i}"].values
        kp[:, i, 1] = df[f"y{i}"].values
    bbox = df[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values.astype(np.float32)
    return kp, bbox


def normalize_kp(kp, bbox):
    wrist = kp[:, 0:1, :].copy()
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    diag = np.sqrt(w * w + h * h) + 1e-6
    scale = diag[:, None, None]
    kp_norm = (kp - wrist) / scale
    denorm = np.concatenate([wrist.squeeze(1), diag[:, None]], axis=1)
    return kp_norm, denorm


def denormalize_kp(kp_norm, denorm):
    wrist_x = denorm[..., 0:1]
    wrist_y = denorm[..., 1:2]
    scale = denorm[..., 2:3]
    out = kp_norm.copy()
    out[..., 0] = kp_norm[..., 0] * scale + wrist_x
    out[..., 1] = kp_norm[..., 1] * scale + wrist_y
    return out


def build_windows_from_range(kp_norm, start, end, T=5, use_velocity=False):
    """在 kp_norm[start:end] 范围内构造滑动窗口。target 帧下标相对于整段序列。"""
    N = kp_norm.shape[0]
    # 段内 feat 需要用整段的 velocity（否则起始帧 velocity=0 不真实）
    flat = kp_norm.reshape(N, -1)
    if use_velocity:
        vel = np.zeros_like(flat)
        vel[1:] = flat[1:] - flat[:-1]
        feat = np.concatenate([flat, vel], axis=1)
    else:
        feat = flat

    # 合法的 target 帧范围：idx 必须满足 idx-T >= 0 且 idx ∈ [start, end)
    X, Y, idx_list = [], [], []
    for i in range(max(start, T), end):
        X.append(feat[i - T:i])
        Y.append(flat[i])
        idx_list.append(i)
    if not X:
        return (np.zeros((0, T, feat.shape[1]), dtype=np.float32),
                np.zeros((0, COORD_DIM), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return (np.stack(X).astype(np.float32),
            np.stack(Y).astype(np.float32),
            np.array(idx_list, dtype=np.int64))


class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X); self.Y = torch.from_numpy(Y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


class LSTMPredictor(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, out_dim=COORD_DIM):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])


def compute_mpjpe_pck(pred_norm, gt_norm, denorm_gt, pck_thresh=0.05):
    pred_kp = pred_norm.reshape(-1, KP_NUM, 2)
    gt_kp = gt_norm.reshape(-1, KP_NUM, 2)
    pred_px = denormalize_kp(pred_kp, denorm_gt)
    gt_px = denormalize_kp(gt_kp, denorm_gt)
    err = np.linalg.norm(pred_px - gt_px, axis=-1)
    mpjpe = err.mean()
    diag = denorm_gt[:, 2:3]
    correct = (err < pck_thresh * diag).mean()
    return float(mpjpe), float(correct)


# ============================================================
# 朴素基线：pred = 上一帧
# ============================================================

def naive_copy_previous(segs, split_plan, use_velocity_unused=False):
    """对 split_plan 里 test 部分的每个样本，用 上一帧关键点 作为预测。"""
    preds, gts, denorms = [], [], []
    for sn, (_, _, test_range) in split_plan.items():
        s = segs[sn]
        kp_norm = s["kp_norm"]; denorm = s["denorm"]
        flat = kp_norm.reshape(len(kp_norm), -1)
        start, end = test_range
        for i in range(max(start, 1), end):
            preds.append(flat[i - 1])  # 上一帧
            gts.append(flat[i])
            denorms.append(denorm[i])
    if not preds:
        return 0.0, 0.0
    preds = np.stack(preds); gts = np.stack(gts); denorms = np.stack(denorms)
    return compute_mpjpe_pck(preds, gts, denorms)


# ============================================================
# 训练一组配置
# ============================================================

def train_one_config(X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
                     D_va, D_te, in_dim, epochs, lr, batch, device, tag, out_dir):
    model = LSTMPredictor(in_dim=in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    tr_loader = DataLoader(SeqDataset(X_tr, Y_tr), batch_size=batch, shuffle=True)
    va_loader = DataLoader(SeqDataset(X_va, Y_va), batch_size=batch, shuffle=False)

    log = []
    best_val_mpjpe = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train(); tr_loss = 0.0; n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0); n += xb.size(0)
        tr_loss /= n

        model.eval(); va_loss = 0.0; n = 0; preds = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                p = model(xb)
                va_loss += crit(p, yb).item() * xb.size(0); n += xb.size(0)
                preds.append(p.cpu().numpy())
        va_loss /= n
        preds = np.concatenate(preds, axis=0)
        va_mpjpe, va_pck = compute_mpjpe_pck(preds, Y_va, D_va)

        log.append({"epoch": ep, "train_mse": tr_loss, "val_mse": va_loss,
                    "val_mpjpe_px": va_mpjpe, "val_pck@0.05": va_pck})
        print(f"[{tag}] ep{ep:03d}  tr={tr_loss:.5f}  va={va_loss:.5f}  "
              f"MPJPE={va_mpjpe:.2f}  PCK={va_pck:.4f}")

        if va_mpjpe < best_val_mpjpe:
            best_val_mpjpe = va_mpjpe
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    te_preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), batch):
            xb = torch.from_numpy(X_te[i:i + batch]).to(device)
            te_preds.append(model(xb).cpu().numpy())
    te_preds = np.concatenate(te_preds, axis=0) if len(X_te) > 0 else np.zeros((0, COORD_DIM))
    te_mpjpe, te_pck = compute_mpjpe_pck(te_preds, Y_te, D_te) if len(X_te) > 0 else (0.0, 0.0)
    print(f"[{tag}] TEST MPJPE={te_mpjpe:.2f}  PCK={te_pck:.4f}")

    pd.DataFrame(log).to_csv(out_dir / f"lstm_{tag}_log.csv", index=False)

    # 测试集逐帧预测
    if len(te_preds) > 0:
        te_pred_kp = denormalize_kp(te_preds.reshape(-1, KP_NUM, 2), D_te)
        te_gt_kp = denormalize_kp(Y_te.reshape(-1, KP_NUM, 2), D_te)
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

    return {"tag": tag, "best_val_mpjpe": best_val_mpjpe,
            "test_mpjpe": te_mpjpe, "test_pck": te_pck, "log": pd.DataFrame(log)}


# ============================================================
# 实验流程
# ============================================================

def run_experiment(segs, split_plan, epochs, lr, batch, device, out_dir, scheme_tag):
    """
    split_plan: dict {seg_name: (train_range, val_range, test_range)}  其中 range=(start,end)
    scheme_tag: "within" 或 "crossseg"
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    # 朴素基线
    n_mpjpe, n_pck = naive_copy_previous(segs, split_plan)
    print(f"\n[{scheme_tag}] 朴素基线(复制上一帧): MPJPE={n_mpjpe:.2f}  PCK={n_pck:.4f}")
    all_rows.append({"config": "naive_copy_previous", "best_val_mpjpe_px": float("nan"),
                     "test_mpjpe_px": n_mpjpe, "test_pck@0.05": n_pck})

    results = {"naive_copy_previous": {"test_mpjpe": n_mpjpe, "test_pck": n_pck}}

    # 两组 LSTM
    for tag, use_vel in [("baseline", False), ("velocity", True)]:
        # 拼接 5 段的 train/val/test
        Xtr, Ytr, _ = [], [], []
        Xva, Yva, Dva = [], [], []
        Xte, Yte, Dte = [], [], []
        for sn, (tr_r, va_r, te_r) in split_plan.items():
            s = segs[sn]
            X, Y, idx = build_windows_from_range(s["kp_norm"], tr_r[0], tr_r[1], T=5, use_velocity=use_vel)
            Xtr.append(X); Ytr.append(Y)
            X, Y, idx = build_windows_from_range(s["kp_norm"], va_r[0], va_r[1], T=5, use_velocity=use_vel)
            Xva.append(X); Yva.append(Y); Dva.append(s["denorm"][idx])
            X, Y, idx = build_windows_from_range(s["kp_norm"], te_r[0], te_r[1], T=5, use_velocity=use_vel)
            Xte.append(X); Yte.append(Y); Dte.append(s["denorm"][idx])

        Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)
        Xva = np.concatenate(Xva); Yva = np.concatenate(Yva); Dva = np.concatenate(Dva)
        Xte = np.concatenate(Xte); Yte = np.concatenate(Yte); Dte = np.concatenate(Dte)

        in_dim = COORD_DIM * (2 if use_vel else 1)
        print(f"\n=== [{scheme_tag}] LSTM {tag}  in_dim={in_dim}  "
              f"train={len(Xtr)} val={len(Xva)} test={len(Xte)} ===")

        r = train_one_config(Xtr, Ytr, Xva, Yva, Xte, Yte, Dva, Dte,
                             in_dim, epochs, lr, batch, device,
                             f"{scheme_tag}_{tag}", out_dir)
        results[tag] = r
        all_rows.append({"config": f"lstm_{tag}",
                         "best_val_mpjpe_px": r["best_val_mpjpe"],
                         "test_mpjpe_px": r["test_mpjpe"],
                         "test_pck@0.05": r["test_pck"]})

    # 训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for tag in ["baseline", "velocity"]:
        log = results[tag]["log"]
        axes[0].plot(log["epoch"], log["train_mse"], label=f"{tag} train")
        axes[0].plot(log["epoch"], log["val_mse"], label=f"{tag} val", linestyle="--")
        axes[1].plot(log["epoch"], log["val_mpjpe_px"], label=tag)
        axes[2].plot(log["epoch"], log["val_pck@0.05"], label=tag)
    axes[1].axhline(n_mpjpe, color="gray", linestyle=":", label=f"naive ({n_mpjpe:.1f}px)")
    axes[2].axhline(n_pck, color="gray", linestyle=":", label=f"naive ({n_pck:.3f})")
    axes[0].set_title(f"MSE Loss ({scheme_tag})"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].set_title(f"Val MPJPE (px) ({scheme_tag})"); axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[2].set_title(f"Val PCK@0.05 ({scheme_tag})"); axes[2].set_xlabel("epoch"); axes[2].legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"training_curves_{scheme_tag}.png", dpi=150)
    plt.close()

    pd.DataFrame(all_rows).to_csv(out_dir / f"summary_{scheme_tag}.csv", index=False)
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    segs = {}
    for name in ["seg1_openclose", "seg2_wave", "seg3_finger", "seg4_point", "seg5_switch"]:
        p = csv_dir / f"{name}.csv"
        kp, bbox = load_seg_csv(p)
        kp_norm, denorm = normalize_kp(kp, bbox)
        segs[name] = {"kp": kp, "bbox": bbox, "kp_norm": kp_norm, "denorm": denorm}
        print(f"[INFO] {name}: {len(kp)} 帧")

    # 方案一：段内划分 70/15/15
    within_plan = {}
    for sn, s in segs.items():
        N = len(s["kp"])
        a = int(N * 0.70)
        b = int(N * 0.85)
        within_plan[sn] = ((0, a), (a, b), (b, N))
    print("\n========================================")
    print("    方案 A：段内划分（within-segment）    ")
    print("========================================")
    run_experiment(segs, within_plan, args.epochs, args.lr, args.batch, device, out_dir, "within")

    # 方案二：跨段划分（保留原来的对比）
    cross_plan = {
        "seg1_openclose": ((0, len(segs["seg1_openclose"]["kp"])), (0, 0), (0, 0)),
        "seg2_wave":      ((0, len(segs["seg2_wave"]["kp"])),      (0, 0), (0, 0)),
        "seg3_finger":    ((0, len(segs["seg3_finger"]["kp"])),    (0, 0), (0, 0)),
        "seg4_point":     ((0, 0), (0, len(segs["seg4_point"]["kp"])), (0, 0)),
        "seg5_switch":    ((0, 0), (0, 0), (0, len(segs["seg5_switch"]["kp"]))),
    }
    print("\n========================================")
    print("    方案 B：跨段划分（cross-segment）    ")
    print("========================================")
    run_experiment(segs, cross_plan, args.epochs, args.lr, args.batch, device, out_dir, "crossseg")

    print(f"\n[DONE] 全部结果保存在 {out_dir}")


if __name__ == "__main__":
    main()
