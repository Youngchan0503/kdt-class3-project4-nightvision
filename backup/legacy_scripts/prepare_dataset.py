import os
import shutil
import random

def prepare_dataset():
    src_dir = r"C:\project4\labels"
    dest_dir = r"C:\project4\dataset_custom_stairs"

    # 이미지 & 텍스트 파일 매칭
    all_files = os.listdir(src_dir)
    jpg_files = [f for f in all_files if f.lower().endswith(".jpg")]
    
    # 짝이 맞는 파일만 필터링
    valid_pairs = []
    for jpg in jpg_files:
        txt = os.path.splitext(jpg)[0] + ".txt"
        if os.path.exists(os.path.join(src_dir, txt)):
            valid_pairs.append((jpg, txt))

    print(f"[*] 총 유효한 (이미지, 라벨) 쌍: {len(valid_pairs)}개")

    # 랜덤 셔플 (고정 시드)
    random.seed(42)
    random.shuffle(valid_pairs)

    # 8:2 분할
    train_count = int(len(valid_pairs) * 0.8)
    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:]

    print(f"[*] Train 데이터: {len(train_pairs)}개")
    print(f"[*] Val 데이터:   {len(val_pairs)}개")

    # 대상 디렉토리 생성
    train_img_dir = os.path.join(dest_dir, "images", "train")
    val_img_dir = os.path.join(dest_dir, "images", "val")
    train_lbl_dir = os.path.join(dest_dir, "labels", "train")
    val_lbl_dir = os.path.join(dest_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Train 파일 복사
    for jpg, txt in train_pairs:
        shutil.copy2(os.path.join(src_dir, jpg), os.path.join(train_img_dir, jpg))
        shutil.copy2(os.path.join(src_dir, txt), os.path.join(train_lbl_dir, txt))

    # Val 파일 복사
    for jpg, txt in val_pairs:
        shutil.copy2(os.path.join(src_dir, jpg), os.path.join(val_img_dir, jpg))
        shutil.copy2(os.path.join(src_dir, txt), os.path.join(val_lbl_dir, txt))

    # data.yaml 생성
    norm_dest_dir = dest_dir.replace('\\', '/')
    yaml_content = f"""path: {norm_dest_dir}
train: images/train
val: images/val

nc: 1
names: ['stairs']
"""
    yaml_path = os.path.join(dest_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[OK] 데이터셋 분할 완료! 생성 경로: {dest_dir}")
    print(f"[OK] data.yaml 생성 완료: {yaml_path}")

if __name__ == "__main__":
    prepare_dataset()
