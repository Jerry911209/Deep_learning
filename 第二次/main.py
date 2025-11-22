# main.py

import os
import torch

# 這個是你新的「切 80:20 成 train_val / test」的檔案
from split_dataset import ensure_trainval_test_split

from data_expansion import run_expansion, need_expansion

# 這個還是沿用你原本的，但要記得在裡面改成讀 data_split
from train_split import create_dataloaders

# 訓練與畫模型比較圖的函式，跟原本一樣
from train_models import train_model, plot_model_comparison

from datetime import datetime

def main():
    DATA_SPLIT_PATH = os.path.join("archive (2)", "data_split")
    # 顯示 CUDA 狀態（純資訊用）
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:  ", torch.version.cuda)

    # 0️⃣ 確保已經把原始資料切成 80:20 → train_val / test
    #    如果之前已經切過，ensure_trainval_test_split() 會自動略過
    print("\n=== 檢查並建立 train_val / test 資料夾結構（80:20） ===")
    # 先檢查你指定的路徑是否已經存在
    if os.path.isdir(DATA_SPLIT_PATH):
        print(f"偵測到 data_split 已存在：{DATA_SPLIT_PATH}")
        print("👉 略過切分，不重複建立 train_val / test。")
    else:
        print(f"找不到 data_split，建立中：{DATA_SPLIT_PATH}")
        ensure_trainval_test_split()  # 執行 80:20 分割

    # 1️⃣ 對 train_val 做 3 倍資料擴充（只做一次）
    print("\n=== 檢查是否需要對 train_val 做資料擴充（每張原圖 3 張） ===")
    if need_expansion():
        print("→ 尚未擴充，開始進行資料擴充...")
        run_expansion()
    else:
        print("→ 偵測到已擴充過，略過資料擴充。")


    # # 1️⃣ 建立 DataLoader（在 train_split.py 裡再把 train_val 切成 Train / Val）
    # print("\n=== 建立 DataLoader 並切 Train / Val / Test ===")
    # train_loader, val_loader, test_loader = create_dataloaders(batch_size=32)

    # 2️⃣ 開始訓練模型
    print("\n=== 開始訓練模型（cnn） ===")
    # 想一次跑三個模型可以改成：
    # models_to_run = ["cnn", "resnet18", "resnet34"]
    models_to_run = ["cnn","cnn_nomal"]

    #超參數定義
    num_epochs=300# 最大 epoch（Early Stopping 會提前停）
    batch_size=32
    lr=3e-4
    seed=42
    patience=30
    
    # 依當下時間與超參數建立實驗根目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_bs{batch_size}_lr{lr}_pat{patience}"
    exp_root = os.path.join("runs", exp_name)
    os.makedirs(exp_root, exist_ok=True)

    print(f"\n本次實驗輸出會放在：{exp_root}\n")
    all_results = {}  # 存每個模型的 metrics

    for name in models_to_run:
        print("\n" + "=" * 60)
        print(f"開始訓練模型：{name}")
        print("=" * 60)

        # train_model 會自己在裡面再呼叫 create_dataloaders
        # 如果你希望用剛剛建好的 train_loader / val_loader / test_loader
        # 也可以把 train_model 改成接 DataLoader 當參數
        metrics, history = train_model(
            model_name=name,
            num_epochs=num_epochs,   # 最大 epoch（Early Stopping 會提前停）
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            patience=patience,
            exp_root=exp_root,
        )

        all_results[name] = metrics

    # 3️⃣ 畫一張模型比較表的圖片
    # #    預設會存在 plots/model_comparison_table.png
    # plot_model_comparison(all_results)
    # 畫一張「三個模型指標比較表格」圖（存到 exp_root）
    plot_model_comparison(all_results, out_dir=exp_root)
    print(f"\n已輸出比較表圖：{os.path.join(exp_root, 'model_comparison_table.png')}")

    # print("\n=== 所有模型訓練完成，結果圖在 plots/ 資料夾 ===")

    # 4️⃣ 終端機文字版比較
    print("\n\n===== 模型在 Test Set 的指標比較 =====")
    for name, m in all_results.items():
        print(f"\n模型：{name}")
        print(f"  Accuracy:            {m['accuracy']:.4f}")
        print(f"  F1-score (macro):    {m['f1_macro']:.4f}")
        print(f"  F1-score (weighted): {m['f1_weighted']:.4f}")

    print("\n比較表輸出完成！")


if __name__ == "__main__":
    main()
