import os
import shutil
import random
from PIL import Image
from torchvision import transforms
from pathlib import Path

# ============================================================
# 取得專案根目錄 Deep_learning/
# （第二次/ 是這個檔案所在，因此 parent 1 層就是 Deep_learning）
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]

# ============================================================
# 使用「相對路徑」：相對於 BASE_DIR，不與執行位置無關
# ============================================================
SRC_ROOT = BASE_DIR / "archive (2)" / "data"
DST_ROOT = BASE_DIR / "archive (2)" / "data_augmented"

# 目標 Gray_Leaf_Spot 增補至多少張
TARGET_NUM_GRAY = 1200

# 圖片副檔名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def copy_original_dataset():
    """把 SRC_ROOT 底下的每個類別資料夾完整複製到 DST_ROOT"""
    print("=== 複製原始資料到新資料夾 ===")
    os.makedirs(DST_ROOT, exist_ok=True)

    for class_name in sorted(os.listdir(SRC_ROOT)):
        src_class_dir = os.path.join(SRC_ROOT, class_name)
        if not os.path.isdir(src_class_dir):
            continue

        dst_class_dir = os.path.join(DST_ROOT, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)

        count = 0
        for fname in os.listdir(src_class_dir):
            if not is_image_file(fname):
                continue
            src_path = os.path.join(src_class_dir, fname)
            dst_path = os.path.join(dst_class_dir, fname)
            shutil.copy2(src_path, dst_path)
            count += 1

        print(f"已複製 {class_name}: {count} 張")


def count_images_per_class(root):
    """統計 root 底下各類別圖片數量"""
    print("\n=== 各類別圖片數量統計（root = {}） ===".format(os.path.abspath(root)))
    total = 0
    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue

        count = 0
        for fname in os.listdir(class_dir):
            if is_image_file(fname):
                count += 1

        print(f"{class_name}: {count}")
        total += count

    print(f"Total images: {total}")


def augment_gray_leaf_spot():
    """
    在 DST_ROOT 裡對 Gray_Leaf_Spot 做資料擴增，
    直到 Gray_Leaf_Spot 的總張數達到 TARGET_NUM_GRAY。
    """
    class_name = "Gray_Leaf_Spot"
    dst_class_dir = os.path.join(DST_ROOT, class_name)

    if not os.path.isdir(dst_class_dir):
        print(f"[警告] 找不到類別資料夾: {dst_class_dir}")
        return

    image_files = [f for f in os.listdir(dst_class_dir) if is_image_file(f)]

    if not image_files:
        print("[警告] Gray_Leaf_Spot 資料夾裡沒有圖片")
        return

    current_num = len(image_files)
    print(f"\nGray_Leaf_Spot 原始圖片數量: {current_num}")

    if current_num >= TARGET_NUM_GRAY:
        print(f"目前數量 {current_num} 已≥目標 {TARGET_NUM_GRAY}，不需擴充。")
        return

    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
    ])

    image_paths = [os.path.join(dst_class_dir, f) for f in image_files]

    new_added = 0
    idx = 0

    existing_names = set(os.listdir(dst_class_dir))

    while current_num < TARGET_NUM_GRAY:
        src_path = random.choice(image_paths)
        try:
            with Image.open(src_path).convert("RGB") as img:
                aug_img = augment_transform(img)

                while True:
                    new_name = f"aug_{idx:05d}.jpg"
                    idx += 1
                    if new_name not in existing_names:
                        existing_names.add(new_name)
                        break

                save_path = os.path.join(dst_class_dir, new_name)
                aug_img.save(save_path)

                new_added += 1
                current_num += 1

                if new_added % 50 == 0:
                    print(f"已新增 {new_added} 張，目前總數: {current_num}")

        except Exception as e:
            print(f"[錯誤] 處理圖片失敗 {src_path}: {e}")

    print(f"\nGray_Leaf_Spot 擴充完成！")
    print(f"新增 {new_added} 張，現在總數 = {current_num}（目標 {TARGET_NUM_GRAY}）")


def run_expansion():
    print("原始資料夾 SRC_ROOT：", os.path.abspath(SRC_ROOT))
    print("新資料夾 DST_ROOT：", os.path.abspath(DST_ROOT))
    print("SRC_ROOT 是否存在？", SRC_ROOT.exists())

    # 1. 複製原資料
    copy_original_dataset()

    # 2. 統計（擴增前）
    print("\n--- 新資料夾（擴增前）統計 ---")
    count_images_per_class(DST_ROOT)

    # 3. 擴增 Gray_Leaf_Spot
    augment_gray_leaf_spot()

    # 4. 統計（擴增後）
    print("\n--- 新資料夾（擴增後）統計 ---")
    count_images_per_class(DST_ROOT)


if __name__ == "__main__":
    run_expansion()
