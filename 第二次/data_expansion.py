# data_expansion.py
#
# 功能：
# - 針對 archive (2)/data_split/train_val 底下的每個類別資料夾
# - 每張「原始圖片」（檔名不以 aug_ 開頭）依序產生 3 張新圖：
#     aug_00001.jpg : 旋轉 +20° + 顏色抖動
#     aug_00002.jpg : 旋轉 -20° + 顏色抖動
#     aug_00003.jpg : 水平翻轉 + 顏色抖動
# - 若資料夾內已經有 aug_*.jpg，視為該類別已做過擴充，會整個略過
#
# 使用方式（例如在 main.py）：
#   from data_expansion import run_expansion, need_expansion
#   ...
#   if need_expansion():
#       run_expansion()

import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

# 你的 split 後資料根目錄
DATA_SPLIT_ROOT = os.path.join("archive (2)", "data_split")
TRAINVAL_ROOT   = os.path.join(DATA_SPLIT_ROOT, "train_val")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def get_class_dirs(root: str):
    """取得 train_val 底下所有類別資料夾 (class_name, full_path) 列表"""
    if not os.path.isdir(root):
        return []
    results = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            results.append((name, p))
    return results


def class_already_augmented(class_dir: str) -> bool:
    """
    判斷這個類別資料夾是不是已經做過擴充：
    - 若裡面存在至少一張檔名以 'aug_' 開頭且為圖片檔，就視為已擴充
    """
    for fname in os.listdir(class_dir):
        if fname.startswith("aug_") and is_image_file(fname):
            return True
    return False


# 顏色抖動（隨機）
color_jitter = transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.05
)

# =========================
# 三種固定的 Augmentation
# =========================
def apply_aug1(img: Image.Image) -> Image.Image:
    """
    aug_00001.jpg：旋轉 +20° + 顏色抖動
    """
    out = F.rotate(img, 20)      # 順時針旋轉 20 度
    out = color_jitter(out)      # 顏色抖動
    return out


def apply_aug2(img: Image.Image) -> Image.Image:
    """
    aug_00002.jpg：旋轉 -20° + 顏色抖動
    """
    out = F.rotate(img, -20)     # 逆時針旋轉 20 度
    out = color_jitter(out)      # 顏色抖動
    return out


def apply_aug3(img: Image.Image) -> Image.Image:
    """
    aug_00003.jpg：水平翻轉 + 顏色抖動
    """
    out = F.hflip(img)           # 水平翻轉
    out = color_jitter(out)      # 顏色抖動
    return out


def next_aug_index(class_dir: str) -> int:
    """
    找出這個資料夾裡現有 aug_XXXXX.jpg 的最大編號，
    回傳下一個可以使用的編號（從 1 開始）。
    這樣就算之後再補做一點點 aug，也不會覆蓋舊檔。
    """
    max_idx = 0
    for fname in os.listdir(class_dir):
        if not fname.startswith("aug_"):
            continue
        if not is_image_file(fname):
            continue
        stem, _ = os.path.splitext(fname)   # aug_00012
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1   # 下一個可用的 index


def augment_fixed_3x_for_class(class_name: str, class_dir: str):
    """
    針對單一類別資料夾：
    - 找出所有「原始圖片」（檔名不以 aug_ 開頭）
    - 每張依序產生 3 張新圖 (aug1, aug2, aug3)
    """
    # 若已做過擴充，整個類別略過（避免越來越多）
    if class_already_augmented(class_dir):
        print(f"[略過] 類別 {class_name} 已有 aug_ 開頭圖片，視為已擴充。")
        return

    # 取得原始圖片列表
    original_files = [
        f for f in os.listdir(class_dir)
        if is_image_file(f) and not f.startswith("aug_")
    ]

    if not original_files:
        print(f"[略過] 類別 {class_name} 沒有原始圖片。")
        return

    print(f"\n=== 類別 {class_name} 擴充開始 ===")
    print(f"路徑：{class_dir}")
    print(f"原始張數：{len(original_files)}")

    # 從現有的 aug_ 編號後面開始接續
    idx = next_aug_index(class_dir)

    for fname in original_files:
        src_path = os.path.join(class_dir, fname)

        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"[錯誤] 無法讀取 {src_path}：{e}")
            continue

        # aug1
        out1 = apply_aug1(img)
        save1 = os.path.join(class_dir, f"aug_{idx:05d}.jpg")
        out1.save(save1)
        idx += 1

        # aug2
        out2 = apply_aug2(img)
        save2 = os.path.join(class_dir, f"aug_{idx:05d}.jpg")
        out2.save(save2)
        idx += 1

        # aug3
        out3 = apply_aug3(img)
        save3 = os.path.join(class_dir, f"aug_{idx:05d}.jpg")
        out3.save(save3)
        idx += 1

    print(f"=== 類別 {class_name} 擴充完成：新增 {len(original_files) * 3} 張 ===")


def run_expansion():
    """
    針對 train_val 底下每個類別資料夾，
    執行「每張原圖 -> 3 張 aug_ 圖」的擴充。
    """
    print("\n=== 對 train_val 做固定 3 倍擴充（每張原圖產生 3 張 aug_*） ===")
    print("train_val 路徑：", os.path.abspath(TRAINVAL_ROOT))

    class_dirs = get_class_dirs(TRAINVAL_ROOT)
    if not class_dirs:
        print("[錯誤] 找不到任何類別資料夾，請確認 split_dataset 有成功切好 train_val。")
        return

    for class_name, class_dir in class_dirs:
        augment_fixed_3x_for_class(class_name, class_dir)

    print("\n=== 所有類別擴充流程結束 ===")


def need_expansion() -> bool:
    """
    只要 train_val 內任一類別已經包含 aug_ 開頭的圖片，
    就視為已擴充過，回傳 False。
    若完全沒有 aug_ 圖 → 回傳 True（需要擴充）
    """
    class_dirs = get_class_dirs(TRAINVAL_ROOT)

    if not class_dirs:
        print("[警告] 找不到 train_val 資料夾或類別資料夾，無法檢查擴充。")
        return False

    for class_name, class_dir in class_dirs:
        for fname in os.listdir(class_dir):
            if fname.startswith("aug_") and is_image_file(fname):
                print(f"[偵測到] {class_name} 已含 aug_ 圖片 → 視為已擴充。")
                return False

    print("[結果] 所有類別都沒有 aug_ 開頭圖片 → 需要擴充！")
    return True


if __name__ == "__main__":
    if need_expansion():
        run_expansion()
    else:
        print("偵測到已擴充過，略過。")
