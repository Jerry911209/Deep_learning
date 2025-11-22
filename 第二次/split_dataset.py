import os
import shutil
import random

# 原始資料根目錄：底下直接是 4 個類別資料夾
SRC_ROOT = os.path.join("archive (2)", "data")

# 新的資料夾：專門放 80:20 切好的 train_val / test
OUT_ROOT      = os.path.join("archive (2)", "data_split")
TRAINVAL_ROOT = os.path.join(OUT_ROOT, "train_val")
TEST_ROOT     = os.path.join(OUT_ROOT, "test")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_RATIO = 0.8   # 80% 給 train_val，20% 給 test
SPLIT_SEED  = 42    # 為了可重現性


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def split_class_folder(class_name: str):
    """
    針對某一個類別（例如 Gray_Leaf_Spot），
    把 SRC_ROOT/class_name 底下的圖片，依 80:20 分到
    train_val/class_name 與 test/class_name。
    """
    src_class_dir = os.path.join(SRC_ROOT, class_name)
    if not os.path.isdir(src_class_dir):
        print(f"[略過] 找不到類別資料夾：{src_class_dir}")
        return

    # 列出所有圖片檔
    files = [
        f for f in os.listdir(src_class_dir)
        if is_image_file(f)
    ]
    if not files:
        print(f"[警告] 類別 {class_name} 沒有圖片，略過。")
        return

    # 固定 seed，讓結果可重現
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(files)

    n_total = len(files)
    n_trainval = int(n_total * SPLIT_RATIO)
    n_test = n_total - n_trainval

    trainval_files = files[:n_trainval]
    test_files     = files[n_trainval:]

    # 準備輸出資料夾
    dst_trainval_dir = os.path.join(TRAINVAL_ROOT, class_name)
    dst_test_dir     = os.path.join(TEST_ROOT, class_name)
    os.makedirs(dst_trainval_dir, exist_ok=True)
    os.makedirs(dst_test_dir, exist_ok=True)

    # 複製 train_val
    for fname in trainval_files:
        src_path = os.path.join(src_class_dir, fname)
        dst_path = os.path.join(dst_trainval_dir, fname)
        shutil.copy2(src_path, dst_path)

    # 複製 test
    for fname in test_files:
        src_path = os.path.join(src_class_dir, fname)
        dst_path = os.path.join(dst_test_dir, fname)
        shutil.copy2(src_path, dst_path)

    print(
        f"類別 {class_name}: 總數 {n_total} → "
        f"train_val {n_trainval}, test {n_test}"
    )


def make_split():
    """
    直接從 SRC_ROOT 讀所有類別，切 80:20 並複製到 data_split。
    若 OUT_ROOT 已存在，這個函式會照樣把檔案複製進去（可能覆蓋舊檔）。
    """
    print("=== 開始做 80:20 資料切分 ===")
    print("原始資料夾：", os.path.abspath(SRC_ROOT))
    print("輸出資料夾：", os.path.abspath(OUT_ROOT))

    os.makedirs(TRAINVAL_ROOT, exist_ok=True)
    os.makedirs(TEST_ROOT, exist_ok=True)

    for class_name in sorted(os.listdir(SRC_ROOT)):
        src_class_dir = os.path.join(SRC_ROOT, class_name)
        if not os.path.isdir(src_class_dir):
            continue
        split_class_folder(class_name)

    print("\n=== 切分完成！總結 ===")
    count_summary(TRAINVAL_ROOT, "train_val")
    count_summary(TEST_ROOT, "test")


def count_summary(root: str, name: str):
    print(f"\n[{name}] {root}")
    total = 0
    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        cnt = sum(
            1 for f in os.listdir(class_dir)
            if is_image_file(f)
        )
        total += cnt
        print(f"  {class_name}: {cnt}")
    print(f"  Total {name}: {total}")


def split_already_done() -> bool:
    """
    簡單檢查：若 train_val & test 都存在且非空，就視為「已切過」。
    這樣 main 裡可以放心呼叫 ensure_trainval_test_split()
    而不會每次都重複複製。
    """
    if not (os.path.isdir(TRAINVAL_ROOT) and os.path.isdir(TEST_ROOT)):
        return False

    # 檢查有沒有至少一個類別資料夾
    def has_data(path):
        for cls in os.listdir(path):
            class_dir = os.path.join(path, cls)
            if os.path.isdir(class_dir):
                if any(is_image_file(f) for f in os.listdir(class_dir)):
                    return True
        return False

    return has_data(TRAINVAL_ROOT) and has_data(TEST_ROOT)


def ensure_trainval_test_split():
    """
    提供給 main.py 使用：
    - 若尚未切好 train_val/test，就做一次 split。
    - 若已經存在切好的結果，就略過。
    """
    if split_already_done():
        print("偵測到 data_split/train_val & test 已存在，略過切分。")
    else:
        make_split()


if __name__ == "__main__":
    # 單獨執行這支檔案時，直接做切分
    make_split()
