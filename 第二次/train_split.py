# train_split.py
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from collections import Counter

DATA_ROOT = os.path.join("archive (2)", "data_augmented")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def create_dataloaders(batch_size=32, seed=42):
    """讀取整個 data_augmented，切 70/10/20，建立 DataLoader，並印出比數與分布。"""

    # 1. 讀資料
    full_dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)

    print("類別對應:", full_dataset.class_to_idx)
    print("資料總數:", len(full_dataset))

    # 2. 切 70/10/20
    total_len = len(full_dataset)
    train_len = int(total_len * 0.7)
    val_len   = int(total_len * 0.1)
    test_len  = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"\n切割結果：")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")
    print(f"Test:  {len(test_dataset)}")

    # 3. 建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. 額外印出每個子資料集中「各類別張數」
    idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}

    def count_classes(subset):
        labels = [full_dataset[i][1] for i in subset.indices]
        counter = Counter(labels)
        return {idx_to_class[k]: v for k, v in counter.items()}

    print("\n=== 每個子資料集類別統計 ===")

    print("\nTrain 分布：")
    print(count_classes(train_dataset))

    print("\nVal 分布：")
    print(count_classes(val_dataset))

    print("\nTest 分布：")
    print(count_classes(test_dataset))

    return train_loader, val_loader, test_loader


# 方便單獨測試這支檔案用
if __name__ == "__main__":
    create_dataloaders()
