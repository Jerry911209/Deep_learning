# train_models.py
#
# 功能總覽：
# 1. 從 train_split.create_dataloaders() 取得 train / val / test DataLoader
# 2. 提供三種模型可選：
#    - "cnn"       -> SimpleCNN4Block（4 個卷積 Block）
#    - "resnet18"  -> torchvision.models.resnet18（預訓練）
#    - "resnet34"  -> torchvision.models.resnet34（預訓練）
# 3. 使用 Adam + lr=1e-3 訓練
# 4. 每個 epoch 記錄 train / val loss & acc，最後畫圖存檔
# 5. 用「驗證損失」做 Early Stopping（patience=30）
# 6. 在 test set 上計算：
#    Accuracy、Precision、Recall、F1-score、混淆矩陣
# 7. main 裡一次跑三個模型，輸出比較表 + 比較圖
# 8. 依照當下時間與超參數建立實驗資料夾，將圖與權重分類存放

import os
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from train_split import create_dataloaders  # 你的 train_split.py 裡的函式

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# =======================
# 1. 自訂 4-Block CNN 模型
# =======================
class SimpleCNN4Block(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN4Block, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 16 -> 16, 256x256 -> 128x128
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
    

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            # Block 2: 16 -> 32 -> 32, 128x128 -> 64x64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

              # Block 3: 32 -> 64, 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 4: 64 -> 128, 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
        )

        # 最後 feature map： [32, 64, 64]
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)            # [B, 32, 64, 64]
        x = x.view(x.size(0), -1)       # [B, 32*64*64]
        x = self.classifier(x)
        return x

class SimpleCNN4Block_nomal(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN4Block_nomal, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 16, 256x256 -> 128x128
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16 -> 32, 128x128 -> 64x64
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32 -> 64, 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 64 -> 128, 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 256 -> 128 -> 64 -> 32 -> 16，所以最後是 [128, 16, 16]
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)          # [B, 128, 16, 16]
        x = x.view(x.size(0), -1)     # [B, 128*16*16]
        x = self.classifier(x)        # [B, num_classes]
        return x
# =======================
# 2. 模型工廠：依名字建立模型
# =======================
def get_model(model_name="cnn", num_classes=4):
    model_name = model_name.lower()

    if model_name == "cnn":
        model = SimpleCNN4Block(num_classes=num_classes)

    elif model_name == "cnn_nomal":
        model = SimpleCNN4Block_nomal(num_classes=num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"未知的 model_name: {model_name}，請用 'cnn'、'resnet18' 或 'resnet34'")

    return model


# =======================
# 3. 訓練 / 驗證一個 epoch
# =======================
def train_one_epoch(model, loader, criterion, optimizer, device):
    # 將模型切換到「訓練模式」
    # 會啟動 Dropout、BatchNorm 的 running mean/var 更新
    model.train()

    running_loss = 0.0   # 用來累積整個 epoch 的 loss
    correct = 0          # 正確預測的數量
    total = 0            # 樣本數總計

    # 從 DataLoader 逐批讀取資料
    for images, labels in loader:
        # 將影像與標籤移到 GPU 或 CPU
        images = images.to(device)
        labels = labels.to(device)

        # 梯度清零（避免前一批資料的梯度累積）
        optimizer.zero_grad()

        # 前向傳遞，得到模型輸出 logits
        outputs = model(images)

        # 計算 loss（如 CrossEntropyLoss）
        loss = criterion(outputs, labels)

        # 反向傳遞，計算每個參數的梯度
        loss.backward()

        # 使用 optimizer 更新模型的參數
        optimizer.step()

        # 累積 loss（loss.item() 是單一 batch 平均 loss，所以乘 batch size）
        running_loss += loss.item() * images.size(0)

        # 取最大 logit 的 index 作為預測類別
        _, preds = torch.max(outputs, 1)

        # 計算這批有多少預測正確
        correct += (preds == labels).sum().item()

        # 累加 batch size（此次有多少資料）
        total += labels.size(0)

    # 計算整個 epoch 的平均 loss 與 accuracy
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    # 切換到「驗證模式」
    # 會關閉 Dropout、BatchNorm 不再更新 running stats
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # 驗證不需要計算梯度 → 減少記憶體與加速
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向傳遞（沒有 backward）
            outputs = model(images)

            # 計算 loss
            loss = criterion(outputs, labels)

            # 累積 loss（乘 batch size 是為了計整個 epoch 的平均 loss）
            running_loss += loss.item() * images.size(0)

            # 計算預測類別
            _, preds = torch.max(outputs, 1)

            # 累計正確數
            correct += (preds == labels).sum().item()

            # 累加總數
            total += labels.size(0)

    # 計算整個 epoch 的平均 loss 與 accuracy
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



# =======================
# 4. 在 test set 上拿 y_true / y_pred
# =======================
def get_test_predictions(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    # 兼容兩種情況：
    # 1) test_loader.dataset 是 Subset(ImageFolder)
    # 2) test_loader.dataset 直接是 ImageFolder
    ds = test_loader.dataset
    if hasattr(ds, "dataset") and hasattr(ds.dataset, "class_to_idx"):
        base_dataset = ds.dataset          # Subset 裡包的 ImageFolder
    else:
        base_dataset = ds                  # 直接就是 ImageFolder

    idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return y_true, y_pred, class_names



# =======================
# 5. 畫訓練過程的 loss / acc 圖（加上早停線）
# =======================
def plot_history(history, model_name, out_dir,
                 best_epoch=None, stop_epoch=None):
    """
    history: dict，含 "train_loss", "val_loss", "train_acc", "val_acc"
    best_epoch: 驗證 loss 最佳的 epoch（早停對應的最佳點）
    stop_epoch: 實際訓練停止的 epoch（觸發 early stopping 的那回合）
    out_dir: 圖片輸出資料夾（會由 train_model 傳進來）
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # ----- Loss 曲線 -----
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")

    if best_epoch is not None:
        plt.axvline(best_epoch, color="red", linestyle="--",
                    label=f"Best Val Loss (e{best_epoch})")
    if stop_epoch is not None and stop_epoch != best_epoch:
        plt.axvline(stop_epoch, color="gray", linestyle=":",
                    label=f"Stop Epoch (e{stop_epoch})")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_loss.png"))
    plt.close()

    # ----- Accuracy 曲線 -----
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")

    if best_epoch is not None:
        plt.axvline(best_epoch, color="red", linestyle="--",
                    label=f"Best Val Loss (e{best_epoch})")
    if stop_epoch is not None and stop_epoch != best_epoch:
        plt.axvline(stop_epoch, color="gray", linestyle=":",
                    label=f"Stop Epoch (e{stop_epoch})")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_acc.png"))
    plt.close()


# =======================
# 6. 畫混淆矩陣
# =======================
def plot_confusion_matrix(y_true, y_pred, class_names, model_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(5, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_cm.png"))
    plt.close()


# =======================
# 7. 主訓練流程（含 Early Stopping）
# =======================
def train_model(
    model_name="cnn",   # 模型種類：'cnn' / 'resnet18' / 'resnet34'
    num_epochs=100,     # 最大訓練輪數
    batch_size=64,
    lr=3e-4,
    seed=42,
    patience=30,
    exp_root="runs",    # 實驗根目錄（外層資料夾，由 main 決定）
    

):
    start_time = time.time()
    """
    exp_root: 一次實驗的根目錄，例如:
      runs/20251117_223045_bs32_lr0.001_pat30
    這裡面會再建 model_name 子資料夾。
    """

    # 對每個模型建立自己的資料夾，例如：
    # runs/時間_bsxx_lrxx_patxx/cnn/

    model_dir = os.path.join(exp_root, model_name)
    plots_dir = os.path.join(model_dir, "plots")
    os.makedirs(model_dir, exist_ok=True)

    # 1️⃣ DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        seed=seed,
    )

    # 2️⃣ 運算裝置 + GPU 資訊
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用裝置:", device)

    if device.type == "cuda":
        print("=== CUDA 設備資訊 ===")
        print("GPU 名稱:", torch.cuda.get_device_name(0))
        print("CUDA 版本:", torch.version.cuda)
        print("cuDNN 版本:", torch.backends.cudnn.version())
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        reserved_mem = torch.cuda.memory_reserved(0) / (1024 ** 3)
        allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        free_mem = total_mem - allocated_mem
        print(f"GPU 總記憶體: {total_mem:.2f} GB")
        print(f"已保留記憶體: {reserved_mem:.2f} GB")
        print(f"已配置記憶體: {allocated_mem:.2f} GB")
        print(f"可用記憶體:   {free_mem:.2f} GB")
        print("===================")
    else:
        print("目前使用 CPU，無法顯示 GPU 詳細資訊。")

    # 3️⃣ 模型 / Loss / Optimizer
    model = get_model(model_name, num_classes=4).to(device)
    print(f"\n使用模型: {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)
    #optimizer = torch.optim.RAdam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = None          # 紀錄 val_loss 最佳的 epoch
    no_improve_count = 0

    print(f"\n啟用 Early Stopping：以驗證損失為準，patience = {patience}\n")
    print("目前環境","BS=",batch_size,"IR=",lr,"\n")

    # 4️⃣ 訓練迴圈
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Early Stopping 判斷
        if val_loss < best_val_loss:
            # 優化成功 → 更新最佳模型紀錄
            best_val_loss = val_loss
            best_state_dict = model.state_dict()# 存模型權重
            best_epoch = epoch + 1      # epoch 從 0 開始，所以 +1
            no_improve_count = 0 #重置
            print(f"  ➜ 驗證損失下降，新 best_val_loss = {best_val_loss:.4f}（epoch {best_epoch}）")
        else:
            # 沒進步 → 次數累加
            no_improve_count += 1
            print(f"  ➜ 驗證損失無進步（連續 {no_improve_count} 次）")
            
            # 若無提升次數達 patience → 觸發 Early Stopping
            if no_improve_count >= patience:
                print(f"\n⚠ 觸發 Early Stopping：驗證損失已連續 {patience} 個 epoch 無進步，停止訓練。\n")
                break # 結束訓練迴圈

# 訓練結束後的實際 epoch 數（可能因 Early Stopping 少於 num_epochs）
    stop_epoch = len(history["train_loss"])

    # 5️⃣ 載入最佳權重
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"已載入驗證損失最佳的模型權重（val_loss = {best_val_loss:.4f}，epoch {best_epoch}）")

    # 6️⃣ Test set 評估
    y_true, y_pred, class_names = get_test_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"\n[Test 指標 - {model_name}]")
    print(f"Accuracy:              {acc:.4f}")
    print(f"Precision (macro):     {precision_macro:.4f}")
    print(f"Recall    (macro):     {recall_macro:.4f}")
    print(f"F1-score  (macro):     {f1_macro:.4f}")
    print(f"Precision (weighted):  {precision_weighted:.4f}")
    print(f"Recall    (weighted):  {recall_weighted:.4f}")
    print(f"F1-score  (weighted):  {f1_weighted:.4f}")

    # 7️⃣ 畫訓練曲線（帶早停線）
    plot_history(
        history,
        model_name,
        out_dir=plots_dir,
        best_epoch=best_epoch,
        stop_epoch=stop_epoch,
    )

    # 8️⃣ 畫混淆矩陣
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        model_name,
        out_dir=plots_dir,
    )

    # 9️⃣ 儲存最佳模型權重（放在 model_dir 底下）
    save_name = os.path.join(model_dir, f"corn_leaf_{model_name}.pth")
    torch.save(model.state_dict(), save_name)
    print(f"模型已儲存為: {save_name}")

    metrics = {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }
    end_time = time.time()
    total_sec = end_time - start_time
    total_min = total_sec / 60

    print(f"\n📌 {model_name} 訓練時間：")
    print(f"   → {total_sec:.2f} 秒 ({total_min:.2f} 分鐘)")

    return metrics, history


# =======================
# 8. 三模型指標比較表（輸出成一張圖）
# =======================
def plot_model_comparison(all_results, out_dir="plots"):
    """
    all_results: dict，key 是 model_name，
                 value 是 train_model 回傳的 metrics dict
    out_dir: 實驗根目錄（例如 runs/時間_bsxx_lrxx_patxx）
    會產生一張表格圖：{out_dir}/model_comparison_table.png
    """
    os.makedirs(out_dir, exist_ok=True)

    models = list(all_results.keys())
    metrics_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    # 準備表格資料
    table_data = [["Metric"] + models]
    for metric in metrics_names:
        row = [metric]
        for m in models:
            row.append(f"{all_results[m][metric]:.4f}")
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(6, 1 + 0.5 * len(table_data)))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(models) + 1)))
    plt.title("Model Comparison on Test Set", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_comparison_table.png"))
    plt.close()


# =======================
# 9. main：一次跑 CNN / ResNet18 / ResNet34
# =======================
if __name__ == "__main__":
    models_to_run = ["cnn", "resnet18", "resnet34"]

    # 共用的超參數（會反映在資料夾名稱）
    batch_size = 64
    lr = 3e-4
    patience = 30
    num_epochs = 100
    seed = 42

    # 依當下時間與超參數建立實驗根目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_bs{batch_size}_lr{lr}_pat{patience}"
    exp_root = os.path.join("runs", exp_name)
    os.makedirs(exp_root, exist_ok=True)

    print(f"\n本次實驗輸出會放在：{exp_root}\n")

    all_results = {}

    for name in models_to_run:
        print("\n" + "=" * 60)
        print(f"開始訓練模型：{name}")
        print("=" * 60)

        metrics, history = train_model(
            model_name=name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            patience=patience,
            exp_root=exp_root,
        )
        all_results[name] = metrics

    print("\n\n===== 三個模型在 Test Set 的指標比較 =====")
    for name, m in all_results.items():
        print(f"\n模型：{name}")
        print(f"  Accuracy:            {m['accuracy']:.4f}")
        print(f"  F1-score (macro):    {m['f1_macro']:.4f}")
        print(f"  F1-score (weighted): {m['f1_weighted']:.4f}")

    # 畫一張「三個模型指標比較表格」圖（存到 exp_root）
    plot_model_comparison(all_results, out_dir=exp_root)
    print(f"\n已輸出比較表圖：{os.path.join(exp_root, 'model_comparison_table.png')}")
