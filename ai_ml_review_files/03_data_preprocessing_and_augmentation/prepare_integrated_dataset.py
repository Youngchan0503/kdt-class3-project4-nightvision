# -*- coding: utf-8 -*-
"""Dataset Integration Script for SafeStep AI

Integrates 3 data sources into dataset_custom_stairs:
1. Existing dataset_custom_stairs (83 images, all ascending_stairs -> class 0)
2. c:\project4\계단 (68 pairs: class 0=ascending, 1=descending)
3. c:\project4\drive-download-20261105T045852Z-1-001 (25 pairs: class 0=ascending, 1=descending)

Total: 176 images & segmentation labels
Classes:
  0: ascending_stairs (오르막 계단)
  1: descending_stairs (내리막 계단)
"""

import os
import shutil
import random
from PIL import Image

def clean_and_integrate():
    base_dir = r"c:\project4"
    dest_dir = os.path.join(base_dir, "dataset_custom_stairs")
    folder_stairs = os.path.join(base_dir, "계단")
    folder_drive = os.path.join(base_dir, "drive-download-20261105T045852Z-1-001")

    train_img_dir = os.path.join(dest_dir, "images", "train")
    val_img_dir = os.path.join(dest_dir, "images", "val")
    train_lbl_dir = os.path.join(dest_dir, "labels", "train")
    val_lbl_dir = os.path.join(dest_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("[1] 기존 dataset_custom_stairs 현황 확인...")
    existing_train_imgs = len(os.listdir(train_img_dir))
    existing_val_imgs = len(os.listdir(val_img_dir))
    print(f"  - 기존 Train: {existing_train_imgs}장, Val: {existing_val_imgs}장 (총 {existing_train_imgs + existing_val_imgs}장)")

    # 1. '계단' 폴더 처리
    print("\n[2] '계단' 폴더 데이터 수집 및 정규화...")
    stairs_pairs = []
    for f in os.listdir(folder_stairs):
        if f.lower().endswith(".jpg"):
            base = os.path.splitext(f)[0]
            txt_path = os.path.join(folder_stairs, base + ".txt")
            img_path = os.path.join(folder_stairs, f)
            if os.path.exists(txt_path):
                # 안전한 영문 파일명 생성
                safe_name = f"stair_k_{base}"
                stairs_pairs.append((img_path, txt_path, safe_name))

    print(f"  - '계단' 유효 쌍: {len(stairs_pairs)}개")

    # 2. 'drive-download' 폴더 처리
    print("\n[3] 'drive-download' 폴더 데이터 수집 및 정규화...")
    drive_pairs = []
    for idx, f in enumerate(sorted(os.listdir(folder_drive))):
        if f.lower().endswith(".jpg"):
            base = os.path.splitext(f)[0]
            txt_path = os.path.join(folder_drive, base + ".txt")
            img_path = os.path.join(folder_drive, f)
            if os.path.exists(txt_path):
                # 파일명 패턴 식별 (불끈거/불켠거, 올라갈때/내려갈때)
                is_dark = "불꺼" in base or "불끈" in base
                is_down = "내려" in base
                tag = f"{'dark' if is_dark else 'lit'}_{'down' if is_down else 'up'}"
                safe_name = f"stair_dd_{tag}_{idx:02d}"
                drive_pairs.append((img_path, txt_path, safe_name))

    print(f"  - 'drive-download' 유효 쌍: {len(drive_pairs)}개")

    new_pairs = stairs_pairs + drive_pairs
    print(f"\n[4] 추가할 새 데이터 총합: {len(new_pairs)}개")

    # 고정 시드로 셔플 (8:2 분할)
    random.seed(42)
    random.shuffle(new_pairs)

    train_count = int(len(new_pairs) * 0.8)
    new_train = new_pairs[:train_count]
    new_val = new_pairs[train_count:]
    print(f"  - 신규 Train 분배: {len(new_train)}개")
    print(f"  - 신규 Val 분배:   {len(new_val)}개")

    def copy_and_verify(pair_list, target_img_dir, target_lbl_dir):
        copied = 0
        for img_src, txt_src, safe_name in pair_list:
            dest_img = os.path.join(target_img_dir, f"{safe_name}.jpg")
            dest_txt = os.path.join(target_lbl_dir, f"{safe_name}.txt")

            # PIL로 이미지 안전 변환 및 저장 (한글 경로 버그 방지)
            try:
                with Image.open(img_src) as img:
                    img.convert("RGB").save(dest_img, "JPEG", quality=95)
            except Exception as e:
                print(f"이미지 복사 실패: {img_src} ({e})")
                continue

            # 라벨 텍스트 복사 (클래스 0 또는 1 정합성 유지)
            try:
                lines = open(txt_src, "r", encoding="utf-8", errors="ignore").read().strip().split("\n")
                valid_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # 0: ascending, 1: descending
                        cls_str = "1" if cls_id == 1 else "0"
                        valid_lines.append(f"{cls_str} " + " ".join(parts[1:]))

                if valid_lines:
                    with open(dest_txt, "w", encoding="utf-8") as f:
                        f.write("\n".join(valid_lines) + "\n")
                    copied += 1
            except Exception as e:
                print(f"라벨 복사 실패: {txt_src} ({e})")

        return copied

    c_train = copy_and_verify(new_train, train_img_dir, train_lbl_dir)
    c_val = copy_and_verify(new_val, val_img_dir, val_lbl_dir)

    print(f"\n[5] 복사 완료: Train {c_train}개, Val {c_val}개 성공!")

    # data.yaml 2개 클래스로 갱신
    yaml_path = os.path.join(dest_dir, "data.yaml")
    norm_dest = dest_dir.replace("\\", "/")
    yaml_content = f"""path: {norm_dest}
train: images/train
val: images/val

nc: 2
names: ['ascending_stairs', 'descending_stairs']
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[6] data.yaml 갱신 완료: {yaml_path}")
    print(yaml_content)

    final_train = len(os.listdir(train_img_dir))
    final_val = len(os.listdir(val_img_dir))
    print("=" * 60)
    print(f"🎉 [성공] 최종 통합 데이터셋 구축 완료!")
    print(f"  - 총 학습(Train) 이미지: {final_train}장")
    print(f"  - 총 검증(Val) 이미지:   {final_val}장")
    print(f"  - 총합 데이터셋 규모:    {final_train + final_val}장")
    print("=" * 60)

if __name__ == "__main__":
    clean_and_integrate()
