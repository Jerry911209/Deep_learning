# main.py

import os
from data_expansion import run_expansion, DST_ROOT, TARGET_NUM_GRAY, is_image_file
from train_split import create_dataloaders
from train_models import train_model, plot_model_comparison   # ★
import torch


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for fname in os.listdir(folder) if is_image_file(fname))


def need_expansion():
    print("=== 檢查資料擴充狀態 ===")

    if not os.path.isdir(DST_ROOT):
        print(f"[需要擴充] 找不到資料夾：{DST_ROOT}")
        return True

    gls_dir = os.path.join(DST_ROOT, "Gray_Leaf_Spot")
    if not os.path.isdir(gls_dir):
        print(f"[需要擴充] 找不到資料夾：{gls_dir}")
        return True

    count = count_images(gls_dir)
    print(f"Gray_Leaf_Spot 目前有 {count} 張圖片")

    if count < TARGET_NUM_GRAY:
        print(f"[需要擴充] Gray_Leaf_Spot < {TARGET_NUM_GRAY}")
        return True

    print("[不需要擴充] Gray_Leaf_Spot 資料已足夠")
    return False


if __name__ == "__main__":
    # 確認一下 CUDA 狀態（純顯示用）
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:  ", torch.version.cuda)

    # 1️⃣ 先處理資料擴充
    if need_expansion():
        print("\n=== 執行資料擴充流程 ===")
        run_expansion()
    else:
        print("\n=== 跳過資料擴充：資料已完整 ===")

    # # 2️⃣ （可選）切資料，看一下各子資料集的比數與類別分布
    # print("\n=== 建立 DataLoader 並切 Train/Val/Test（僅顯示資訊） ===")
    # _train_loader, _val_loader, _test_loader = create_dataloaders(batch_size=32)

    # 3️⃣ 在這裡呼叫 train_model，跑三個模型
    print("\n=== 開始訓練三個模型（cnn / resnet18 / resnet34） ===")

    models_to_run = ["cnn", "resnet18", "resnet34"]

    all_results = {}  # 這裡會直接存每個 model 的 metrics dict
    for name in models_to_run:
        print("\n" + "=" * 60)
        print(f"開始訓練模型：{name}")
        print("=" * 60)

        # train_model 回傳 (metrics, history)
        metrics, history = train_model(
            model_name=name,
            num_epochs=100,   # 最大 epoch（Early Stopping 會提前停）
            batch_size=256,
            lr=1e-3,
            seed=42,
            patience=20,
        )

        # ✅ 關鍵：這裡直接存 metrics，而不是 {"metrics": metrics}
        all_results[name] = metrics

    # 4️⃣ 畫一張三個模型的指標比較表圖
    #    會在 plots/model_comparison_table.png
    plot_model_comparison(all_results)

    print("\n=== 所有模型訓練完成，結果圖在 plots/ 資料夾 ===")

    # 5️⃣ 終端機文字版比較
    print("\n\n===== 三個模型在 Test Set 的指標比較 =====")
    for name, m in all_results.items():
        print(f"\n模型：{name}")
        print(f"  Accuracy:            {m['accuracy']:.4f}")
        print(f"  F1-score (macro):    {m['f1_macro']:.4f}")
        print(f"  F1-score (weighted): {m['f1_weighted']:.4f}")

    print("\n比較表輸出完成！")

