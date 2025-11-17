#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
普通 MLP（多層感知器）+ 5 折交叉驗證（每折 300 次，逐 epoch 指標 + 圖表輸出 + 20% 測試集）
------------------------------------------------------------------------------------
本版特色：
1) 讀取 CSV（預設最後一欄為標籤；若有 "label" 欄位則優先使用），自動判定「分類/迴歸」
2) 先切出 20% 當「最終測試集」；剩下 80% 用 5-fold 交叉驗證（訓練/驗證）
3) 逐 epoch 印出訓練/驗證指標：
   - 分類：train_loss、train_acc、val_loss、val_acc
   - 迴歸：train_loss(MSE)、train_RMSE、val_loss(MSE)、val_RMSE
4) 每折會繪製「兩個子圖」的圖表（上：Loss；下：Accuracy/RMSE），存在 results/ 資料夾
5) 設定中文字體（Windows 建議使用微軟正黑體），避免圖上中文字亂碼
6) 5 折完成後，使用 80% 訓練資料重新訓練一個模型，最後在 20% 測試集上做最終評估
7) 另輸出 summary.png（各折的最佳驗證指標柱狀圖 + 平均線）

用法：
    - 直接修改檔案上方的 DATA_PATH
    - python d_fixed.py
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

# ========================= 使用者可調整區 =========================
# 建議 Windows 本機：把路徑改成 r"C:\Users\你的名字\Desktop\dataset.csv"
# 若在此環境，請用 /mnt/data/dataset.csv
DATA_PATH = r"dataset.csv"    # ← 修改為你的資料路徑

# 訓練設定
EPOCHS = 300
KFOLDS = 5
BATCH_SIZE = 32
LR = 1e-3
HIDDEN1 = 128
HIDDEN2 = 64
DROPOUT = 0.1
SEED = 42

# 若想強制裝置，可設為 "cuda" 或 "cpu"；設為 None 則自動偵測
FORCE_DEVICE = None

# 圖片輸出設定
RESULT_DIR = "results"  # 圖片輸出資料夾
DPI = 140               # 輸出圖片解析度
# ===============================================================

# 設定中文字體（Windows 建議用微軟正黑體）；避免負號變成方塊
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


# -----------------------------
# MLP 模型定義
# -----------------------------
class MLP(nn.Module):
    """
    三層 MLP：
    Linear(in_dim -> hidden1) -> ReLU -> Dropout ->
    Linear(hidden1 -> hidden2) -> ReLU -> Dropout ->
    Linear(hidden2 -> out_dim)

    - 分類任務：out_dim = 類別數（二分類也使用 2，配合 CrossEntropyLoss）
    - 迴歸任務：out_dim = 1
    """
    def __init__(self, in_dim: int, out_dim: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 資料讀取與任務判定
# -----------------------------
def load_dataset(csv_path: str, target_col: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """讀入 CSV，回傳 (X_df, y_series)。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到資料檔：{csv_path}")
    df = pd.read_csv(csv_path)

    # 標籤欄位：優先用 'label'，否則最後一欄
    if target_col is None:
        target_col = "label" if "label" in df.columns else df.columns[-1]
    if target_col not in df.columns:
        raise ValueError(f"標籤欄 '{target_col}' 不存在。現有欄位：{list(df.columns)}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 對非數值特徵做 one-hot（示範性質；高基數欄位可改更進階做法）
    X = pd.get_dummies(X, drop_first=True)

    # 若標籤為字串類別，轉為整數 id
    if y.dtype == object:
        y = y.astype("category").cat.codes

    return X, y


def infer_task_type(y: pd.Series, max_classes: int = 20) -> Tuple[str, int]:
    """
    任務型態判定：
    - y 為整數且獨特值數量 <= max_classes -> 'classification'
    - 否則 -> 'regression'
    回傳 (task, num_classes)；若為迴歸則 num_classes = 1。
    """
    y_values = pd.to_numeric(y, errors="coerce")
    unique_vals = pd.unique(y_values.dropna())
    is_integer_like = np.all(np.equal(np.mod(unique_vals, 1), 0))
    num_classes = len(unique_vals)

    if is_integer_like and num_classes <= max_classes:
        return "classification", (2 if num_classes <= 2 else num_classes)
    else:
        return "regression", 1


def describe_target(y: pd.Series, task: str):
    """輸出標籤/目標統計：分類→分佈；迴歸→min/max/mean。"""
    print("\n======== 目標欄統計 ========")
    if task == "classification":
        y_np = pd.to_numeric(y, errors="coerce").astype(int).values
        vals, counts = np.unique(y_np, return_counts=True)
        total = counts.sum()
        for v, c in zip(vals, counts):
            print(f"Class {v}: {c} ({c/total:.2%})")
    else:
        y_np = pd.to_numeric(y, errors="coerce").values
        print(f"min={np.nanmin(y_np):.6f} | max={np.nanmax(y_np):.6f} | mean={np.nanmean(y_np):.6f}")
    print("================================\n")


# -----------------------------
# 訓練 / 驗證（逐 epoch）
# -----------------------------
def run_one_epoch(model, dataloader, device, task: str, criterion=None, optimizer=None) -> Tuple[float, float]:
    """
    執行一個 epoch：
    - 若提供 optimizer，執行訓練；否則為驗證。
    - 回傳 (avg_loss, metric)，其中 metric=acc(分類) 或 RMSE(迴歸)。
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses, y_true_all, y_pred_all = [], [], []

    for xb, yb in dataloader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)

        if task == "classification":
            loss = criterion(logits, yb.long())
            preds = torch.argmax(logits, dim=1)
        else:
            preds = logits.squeeze()
            loss = criterion(preds, yb.float())

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.detach().item())
        y_true_all.append(yb.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())

    avg_loss = float(np.mean(losses))
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    if task == "classification":
        metric = float(accuracy_score(y_true, y_pred))
    else:
        metric = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # RMSE

    return avg_loss, metric


def train_one_fold_verbose(model, optimizer, criterion, train_loader, val_loader, device, epochs: int, task: str, fold_idx: int):
    """
    訓練單一折，逐 epoch 印出指標並保存曲線資料以利繪圖。
    - 本函式一開始就把模型搬到 device，避免 CPU/CUDA 混用錯誤。
    回傳：
      - metrics: 該折最佳權重下的驗證結果（分類：accuracy/F1；迴歸：MSE/MAE）
      - curves: 供繪圖使用的字典（train_loss, val_loss, train_acc/RMSE, val_acc/RMSE）
    """
    # ★ 關鍵修正：確保模型在正確裝置上（GPU 或 CPU）
    model.to(device)

    best_metric = -np.inf if task == "classification" else np.inf
    best_state = None

    # 紀錄曲線
    curves = {
        "train_loss": [],
        "val_loss": [],
        "train_metric": [],  # acc 或 RMSE
        "val_metric": []
    }

    for epoch in range(1, epochs + 1):
        tr_loss, tr_metric = run_one_epoch(model, train_loader, device, task, criterion=criterion, optimizer=optimizer)
        with torch.no_grad():
            val_loss, val_metric = run_one_epoch(model, val_loader, device, task, criterion=criterion, optimizer=None)

        curves["train_loss"].append(tr_loss)
        curves["val_loss"].append(val_loss)
        curves["train_metric"].append(tr_metric)
        curves["val_metric"].append(val_metric)

        if task == "classification":
            print(f"[Fold {fold_idx}][Epoch {epoch:03d}/{epochs}] "
                  f"train_loss={tr_loss:.4f} acc={tr_metric:.4f} | "
                  f"val_loss={val_loss:.4f} acc={val_metric:.4f}")
            metric_for_select = val_metric     # 用驗證 accuracy 選最佳
            is_better = metric_for_select > best_metric
        else:
            print(f"[Fold {fold_idx}][Epoch {epoch:03d}/{epochs}] "
                  f"train_loss(MSE)={tr_loss:.6f} RMSE={tr_metric:.6f} | "
                  f"val_loss(MSE)={val_loss:.6f} RMSE={val_metric:.6f}")
            metric_for_select = val_loss       # 用驗證 MSE 選最佳
            is_better = metric_for_select < best_metric

        if is_better:
            best_metric = metric_for_select
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 載回最佳權重
    if best_state is not None:
        model.load_state_dict(best_state)

    # 最終驗證（用最佳權重）
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            if task == "classification":
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(yb.cpu().numpy().tolist())
            else:
                preds = logits.squeeze().cpu().numpy()
                y_pred.extend(np.atleast_1d(preds).tolist())
                y_true.extend(yb.cpu().numpy().tolist())

    if task == "classification":
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        metrics = {"accuracy": acc, "f1_weighted": f1}
    else:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        metrics = {"mse": mse, "mae": mae}

    return metrics, curves


# -----------------------------
# 繪圖工具
# -----------------------------
def plot_fold_curves(curves: Dict[str, List[float]], task: str, fold_idx: int, out_dir: str, dpi: int = DPI):
    """
    產生每折的學習曲線圖：
    - 子圖 1（上）：train_loss 與 val_loss
    - 子圖 2（下）：分類顯示 accuracy；迴歸顯示 RMSE
    檔名：results/fold{fold_idx}_curves.png
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    epochs = np.arange(1, len(curves["train_loss"]) + 1)

    # 上：Loss
    axes[0].plot(epochs, curves["train_loss"], label="train_loss")
    axes[0].plot(epochs, curves["val_loss"], label="val_loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Fold {fold_idx} - Loss 曲線", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    # 下：Accuracy 或 RMSE
    if task == "classification":
        axes[1].plot(epochs, curves["train_metric"], label="train_acc")
        axes[1].plot(epochs, curves["val_metric"], label="val_acc")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"Fold {fold_idx} - Accuracy 曲線", fontsize=12)
    else:
        axes[1].plot(epochs, curves["train_metric"], label="train_RMSE")
        axes[1].plot(epochs, curves["val_metric"], label="val_RMSE")
        axes[1].set_ylabel("RMSE")
        axes[1].set_title(f"Fold {fold_idx} - RMSE 曲線", fontsize=12)

    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    out_path = os.path.join(out_dir, f"fold{fold_idx}_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[PLOT] 已輸出：{out_path}")


def plot_summary_bar(best_scores: List[float], task: str, out_dir: str, dpi: int = DPI):
    """
    產生各折最佳驗證分數的柱狀圖與平均線：results/summary.png
    - 分類：使用驗證 accuracy（越高越好）
    - 迴歸：使用驗證 RMSE（越低越好）
    """
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 5))

    x = np.arange(1, len(best_scores) + 1)
    plt.bar(x, best_scores)
    mean_val = float(np.mean(best_scores))
    plt.axhline(mean_val, linestyle="--", label=f"mean={mean_val:.4f}")

    if task == "classification":
        plt.ylabel("Best Val Accuracy")
        plt.title("各折最佳驗證 Accuracy", fontsize=12)
    else:
        plt.ylabel("Best Val RMSE")
        plt.title("各折最佳驗證 RMSE（越低越好）", fontsize=12)

    plt.xlabel("Fold")
    plt.legend()
    out_path = os.path.join(out_dir, "summary.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[PLOT] 已輸出：{out_path}")


# -----------------------------
# 主流程
# -----------------------------
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 讀資料 + 任務判定 + 目標統計
    X_df, y_series = load_dataset(DATA_PATH, target_col=None)
    print("[INFO] 讀到資料：", X_df.shape[0], "筆樣本，", X_df.shape[1], "個特徵")
    print(X_df.head(3))

    task, num_classes = infer_task_type(y_series)
    print(f"[INFO] 任務型態：{task} | 輸出維度：{num_classes}")
    describe_target(y_series, task)

    # 轉 numpy
    X_all = X_df.values.astype(np.float32)
    y_all = pd.to_numeric(y_series, errors="coerce").values

    # 切 20% 測試集（分類使用 stratify，避免類別不平衡）
    strat = y_all if task == "classification" else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=strat
    )
    print(f"[SPLIT] 訓練+驗證：{X_trainval.shape[0]}，測試：{X_test.shape[0]}")

    # 決定裝置
    device = FORCE_DEVICE if FORCE_DEVICE in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用裝置：{device}")

    # 在 80% trainval 做 5-fold CV
    if task == "classification":
        splitter = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
        split_iter = splitter.split(X_trainval, y_trainval)
        out_dim = num_classes
    else:
        splitter = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
        split_iter = splitter.split(X_trainval)
        out_dim = 1

    os.makedirs(RESULT_DIR, exist_ok=True)

    fold_metrics = []
    val_best_for_plot = []   # summary.png 用（分類：best val acc；迴歸：best val RMSE）

    for fold_idx, idxs in enumerate(split_iter, start=1):
        if task == "classification":
            train_idx, val_idx = idxs
        else:
            train_idx, val_idx = idxs

        print(f"\n================ Fold {fold_idx}/{KFOLDS} ================")
        X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]

        # 每折內標準化（只用訓練集 fit，避免洩漏）
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)

        # DataLoader（對 CUDA 設定 pin_memory 可提升搬運效率）
        pinmem = (device == "cuda")
        X_tr_t, X_val_t = torch.from_numpy(X_tr), torch.from_numpy(X_val)
        if task == "classification":
            y_tr_t = torch.from_numpy(y_tr.astype(np.int64))
            y_val_t = torch.from_numpy(y_val.astype(np.int64))
        else:
            y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
            y_val_t = torch.from_numpy(y_val.astype(np.float32))

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True, pin_memory=pinmem)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False, pin_memory=pinmem)

        # 模型/優化器/損失
        model = MLP(in_dim=X_trainval.shape[1], out_dim=out_dim, hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

        # 訓練一折（帶曲線記錄；函式內部會 model.to(device)）
        t0 = time.time()
        metrics, curves = train_one_fold_verbose(model, optimizer, criterion, train_loader, val_loader, device, EPOCHS, task, fold_idx)
        print(f"[Fold {fold_idx}] 完成，耗時 {time.time() - t0:.1f}s。驗證結果：{metrics}")
        fold_metrics.append(metrics)

        # 摘錄 summary 用的最佳驗證指標
        if task == "classification":
            # 取該折歷程中的最佳 val_acc
            best_val_acc = float(np.max(curves["val_metric"]))
            val_best_for_plot.append(best_val_acc)
        else:
            # 迴歸取最小的 val_RMSE
            best_val_rmse = float(np.min(curves["val_metric"]))
            val_best_for_plot.append(best_val_rmse)

        # 繪圖：每折一張（兩個子圖：Loss + Acc/RMSE）
        plot_fold_curves(curves, task, fold_idx, RESULT_DIR, dpi=DPI)

    # 產生 summary.png（bar + 平均線）
    plot_summary_bar(val_best_for_plot, task, RESULT_DIR, dpi=DPI)

    # ============ 最終測試 ============
    print("\n======== 重新訓練並於 20% 測試集評估 ========")
    scaler_all = StandardScaler()
    X_trainval_scaled = scaler_all.fit_transform(X_trainval).astype(np.float32)
    X_test_scaled = scaler_all.transform(X_test).astype(np.float32)

    pinmem = (device == "cuda")
    X_tr_all_t = torch.from_numpy(X_trainval_scaled)
    X_te_t = torch.from_numpy(X_test_scaled)
    if task == "classification":
        y_tr_all_t = torch.from_numpy(y_trainval.astype(np.int64))
        y_te_t = torch.from_numpy(y_test.astype(np.int64))
    else:
        y_tr_all_t = torch.from_numpy(y_trainval.astype(np.float32))
        y_te_t = torch.from_numpy(y_test.astype(np.float32))

    train_all_loader = DataLoader(TensorDataset(X_tr_all_t, y_tr_all_t), batch_size=BATCH_SIZE, shuffle=True, pin_memory=pinmem)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=BATCH_SIZE, shuffle=False, pin_memory=pinmem)

    model_final = MLP(in_dim=X_trainval.shape[1], out_dim=(num_classes if task == "classification" else 1),
                      hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT)
    optimizer_final = torch.optim.Adam(model_final.parameters(), lr=LR)
    criterion_final = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    model_final.to(device)

    # 簡單訓練 EPOCHS 次（無早停，示範用）
    for ep in range(1, EPOCHS + 1):
        model_final.train()
        for xb, yb in train_all_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model_final(xb)
            loss = criterion_final(logits, yb.long()) if task == "classification" else criterion_final(logits.squeeze(), yb.float())
            optimizer_final.zero_grad()
            loss.backward()
            optimizer_final.step()

    # 在 20% 測試集評估
    model_final.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model_final(xb)
            if task == "classification":
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(yb.cpu().numpy().tolist())
            else:
                preds = logits.squeeze().cpu().numpy()
                y_pred.extend(np.atleast_1d(preds).tolist())
                y_true.extend(yb.cpu().numpy().tolist())

    if task == "classification":
        test_acc = accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, average="weighted")
        print(f"[TEST] Accuracy={test_acc:.4f} | F1(weighted)={test_f1:.4f}")
    else:
        test_mse = mean_squared_error(y_true, y_pred)
        test_mae = mean_absolute_error(y_true, y_pred)
        print(f"[TEST] MSE={test_mse:.6f} | MAE={test_mae:.6f}")

    print("\n[INFO] 全流程完成，圖表已輸出至資料夾：", os.path.abspath(RESULT_DIR))


if __name__ == "__main__":
    main()
