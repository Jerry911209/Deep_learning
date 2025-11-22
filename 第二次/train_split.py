# train_split.py

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from collections import Counter

# 使用 data_split（已經切好的 train_val / test）
DATA_SPLIT_ROOT = os.path.join("archive (2)", "data_split")
TRAINVAL_ROOT   = os.path.join(DATA_SPLIT_ROOT, "train_val")
TEST_ROOT       = os.path.join(DATA_SPLIT_ROOT, "test")

# 圖片轉換
transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


def create_dataloaders(batch_size=32, seed=42):
    """
    讀取 train_val → 切成 Train / Val（8:2）
    讀取 test → 直接當 test
    建立 DataLoader 並印出各子集的分布情況與樣本數比較表
    """

    # 1️⃣ 讀取 train_val 資料
    trainval_dataset = datasets.ImageFolder(TRAINVAL_ROOT, transform=transform)
    print("類別對應:", trainval_dataset.class_to_idx)
    print("train_val 總數:", len(trainval_dataset))

    # 2️⃣ 切 Train / Val = 80% / 20%
    total_len = len(trainval_dataset)
    train_len = int(total_len * 0.8)
    val_len   = total_len - train_len

    train_dataset, val_dataset = random_split(
        trainval_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"\n切割結果：")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")

    # 3️⃣ 讀取 Test（不切）
    test_dataset = datasets.ImageFolder(TEST_ROOT, transform=transform)
    print(f"Test 總數: {len(test_dataset)}")

    # 4️⃣ 建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5️⃣ 類別統計（不重新讀圖 → 使用 targets 或 samples）
    print("\n=== 每個子資料集類別統計 ===")

    # 用 trainval_dataset 的 class_to_idx 建反查表
    idx_to_class = {v: k for k, v in trainval_dataset.class_to_idx.items()}

    def fast_count(dataset, indices=None):
        """
        快速統計各類別張數：
        - 優先使用 dataset.targets（新版本 torchvision）
        - 若沒有 targets，則從 dataset.samples 取 label（不讀圖片，只看 metadata）
        """
        if hasattr(dataset, "targets"):
            all_labels = dataset.targets
        else:
            # ImageFolder 舊版：labels 在 samples 中的第 2 個元素
            all_labels = [s[1] for s in dataset.samples]

        if indices is None:
            labels = all_labels
        else:
            labels = [all_labels[i] for i in indices]

        counter = Counter(labels)
        return {idx_to_class[k]: v for k, v in counter.items()}

    # Train / Val 用同一個 trainval_dataset + indices
    train_counts = fast_count(trainval_dataset, train_dataset.indices)
    val_counts   = fast_count(trainval_dataset, val_dataset.indices)
    # Test 是獨立的 ImageFolder，但 class_to_idx 應該一致，所以同樣用 idx_to_class
    test_counts  = fast_count(test_dataset, None)

    print("\nTrain 分布：")
    print(train_counts)

    print("\nVal 分布：")
    print(val_counts)

    print("\nTest 分布：")
    print(test_counts)

    # 6️⃣ 樣本比數比較表（Train / Val / Test 放在一起比）
    print("\n=== Train / Val / Test 樣本比數比較表（每個類別的張數） ===")

    # 收集所有出現過的類別名稱（避免某個 split 沒出現那個類別）
    all_classes = sorted(set(list(train_counts.keys()) +
                             list(val_counts.keys()) +
                             list(test_counts.keys())))

    # 印表頭
    print(f"{'Class':20s} {'Train':>8s} {'Val':>8s} {'Test':>8s} {'Total':>8s}")
    print("-" * 60)

    for cls in all_classes:
        n_train = train_counts.get(cls, 0)
        n_val   = val_counts.get(cls, 0)
        n_test  = test_counts.get(cls, 0)
        n_total = n_train + n_val + n_test
        print(f"{cls:20s} {n_train:8d} {n_val:8d} {n_test:8d} {n_total:8d}")

    print("-" * 60)
    total_train = sum(train_counts.values())
    total_val   = sum(val_counts.values())
    total_test  = sum(test_counts.values())
    total_all   = total_train + total_val + total_test
    print(f"{'TOTAL':20s} {total_train:8d} {total_val:8d} {total_test:8d} {total_all:8d}")

    return train_loader, val_loader, test_loader


# 方便單獨測試這支檔案用
if __name__ == "__main__":
    create_dataloaders()
