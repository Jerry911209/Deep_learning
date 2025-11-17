import os
import shutil
import random
from PIL import Image
from torchvision import transforms

# === 這裡依照你的實際路徑調整 ===
# 原始資料根目錄（裡面直接放 4 個類別資料夾）
SRC_ROOT = os.path.join("archive (2)", "data")

# 新的「大資料夾」，用來放：原圖 + 擴充後圖片
DST_ROOT = os.path.join("archive (2)", "data_augmented")

# 目標：Gray_Leaf_Spot 在新資料夾中想要的總張數
TARGET_NUM_GRAY = 1200   # 你可以改成你想要的數量

# 認定為圖片的副檔名
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
    """統計某個根目錄底下，各類別資料夾的圖片數量"""
    print("\n=== 各類別圖片數量統計（root = {}） ===".format(os.path.abspath(root)))
    total = 0
    class_counts = {}

    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue

        count = 0
        for fname in os.listdir(class_dir):
            if is_image_file(fname):
                count += 1

        class_counts[class_name] = count
        total += count
        print(f"{class_name}: {count}")

    print(f"Total images: {total}")
    return class_counts


def augment_gray_leaf_spot():
    """
    在新的資料夾 DST_ROOT 裡，對 Gray_Leaf_Spot 做資料擴充，
    直到 Gray_Leaf_Spot 的總張數達到 TARGET_NUM_GRAY。
    """
    class_name = "Gray_Leaf_Spot"
    dst_class_dir = os.path.join(DST_ROOT, class_name)

    if not os.path.isdir(dst_class_dir):
        print(f"[警告] 找不到類別資料夾: {dst_class_dir}")
        return

    # 取得目前 Gray_Leaf_Spot 已存在的圖片（複製過來的原圖）
    image_files = [
        f for f in os.listdir(dst_class_dir)
        if is_image_file(f)
    ]

    if not image_files:
        print("[警告] Gray_Leaf_Spot 資料夾裡沒有圖片")
        return

    current_num = len(image_files)
    print(f"\nGray_Leaf_Spot 在新資料夾中的原始圖片數量: {current_num}")

    if current_num >= TARGET_NUM_GRAY:
        print(f"目前數量 {current_num} 已經 ≥ 目標 {TARGET_NUM_GRAY}，不用擴充。")
        return

    # 擴充使用的 transform（保持為 PIL Image，不轉 tensor）
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

    # 原圖路徑（從新資料夾讀即可）
    image_paths = [os.path.join(dst_class_dir, f) for f in image_files]

    new_added = 0
    # 為避免檔名衝突，找一個起始 index（例如現有檔案數）
    idx = 0

    # 先確保不會覆蓋到已存在的檔名
    existing_names = set(os.listdir(dst_class_dir))
    while current_num < TARGET_NUM_GRAY:
        src_path = random.choice(image_paths)

        try:
            with Image.open(src_path).convert("RGB") as img:
                aug_img = augment_transform(img)

                # 產生一個新的檔名，不與現有檔案重複
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
                    print(f"已新增 {new_added} 張增強圖片，目前 Gray_Leaf_Spot 總數: {current_num}")

        except Exception as e:
            print(f"[錯誤] 處理圖片失敗 {src_path}: {e}")
            continue

    print(f"\nGray_Leaf_Spot 資料擴充完成！")
    print(f"總共新增 {new_added} 張，現在總數 = {current_num}（目標 {TARGET_NUM_GRAY}）")

def run_expansion():
    print("原始資料夾 SRC_ROOT：", os.path.abspath(SRC_ROOT))
    print("新資料夾 DST_ROOT：", os.path.abspath(DST_ROOT))

    # 1. 複製原始資料
    copy_original_dataset()

    # 2. 統計（擴充前）
    print("\n--- 新資料夾（擴充前）統計 ---")
    count_images_per_class(DST_ROOT)

    # 3. 擴充 Gray_Leaf_Spot
    augment_gray_leaf_spot()

    # 4. 統計（擴充後）
    print("\n--- 新資料夾（擴充後）統計 ---")
    count_images_per_class(DST_ROOT)


if __name__ == "__main__":
    run_expansion()

# if __name__ == "__main__":
#     print("原始資料夾 SRC_ROOT：", os.path.abspath(SRC_ROOT))
#     print("新資料夾 DST_ROOT：", os.path.abspath(DST_ROOT))

#     # 1️⃣ 先複製原本資料到新的大資料夾
#     copy_original_dataset()

#     # 2️⃣ 統計新資料夾中各類別圖片數量（擴充前）
#     print("\n--- 新資料夾（擴充前）統計 ---")
#     count_images_per_class(DST_ROOT)

#     # 3️⃣ 在新資料夾中對 Gray_Leaf_Spot 做資料擴充
#     augment_gray_leaf_spot()

#     # 4️⃣ 再統計一次（擴充後）
#     print("\n--- 新資料夾（擴充後）統計 ---")
#     count_images_per_class(DST_ROOT)
