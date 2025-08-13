#!/usr/bin/env python3
"""
从 datasets_trash 抽样并合成带YOLO框的图片。
支持两种模式：
- 按标签（默认）：每个标签各抽样K张，输出到 trashimg/{class_name}/
- 按来源：每个来源各抽样K张，输出到 trashimg/{source}/

目录结构（输入与输出）:
- 输入: <trash_dir>/images/*.jpg(png,jpeg,bmp)
- 输入: <trash_dir>/labels/*.txt (YOLO格式: class cx cy w h)
- 输出: <trash_dir>/trashimg/{group}/<orig_name>.jpg

来源(source)识别规则（按优先级）:
- 文件名包含: "coco" -> coco
- 文件名包含: "cityscapes" -> cityscapes
- 文件名包含: "val" -> validation
- 文件名包含: "train" -> training
- 其他 -> other
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Optional

import cv2
import sys

# 允许从项目根目录导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from config import COCO_CLASSES  # noqa: E402

SUPPORTED_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def detect_source(name: str) -> str:
    lower = name.lower()
    if "coco" in lower:
        return "coco"
    if "cityscapes" in lower:
        return "cityscapes"
    if "val" in lower:
        return "validation"
    if "train" in lower:
        return "training"
    return "other"


essential_colors = {
    0: (255, 0, 0),   # person -> blue (BGR)
    2: (0, 0, 255),   # car -> red (BGR)
}


def draw_yolo_boxes(img_path: Path, label_path: Path, out_path: Path, class_names: List[str]) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]

    if not label_path.exists():
        return False

    try:
        with label_path.open('r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return False

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            cx = float(parts[1]) * w
            cy = float(parts[2]) * h
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h
        except Exception:
            continue

        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(w, cx + bw / 2))
        y2 = int(min(h, cy + bh / 2))

        color = essential_colors.get(cls_id, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
        cv2.putText(img, name, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 统一保存为jpg
    return bool(cv2.imwrite(str(out_path.with_suffix('.jpg')), img))


def collect_images_by_source(images_dir: Path) -> Dict[str, List[Path]]:
    by_source: Dict[str, List[Path]] = {}
    for ext in SUPPORTED_EXTS:
        for img_path in images_dir.glob(ext):
            src = detect_source(img_path.name)
            by_source.setdefault(src, []).append(img_path)
    return by_source


def parse_class_ids(arg: Optional[str]) -> Optional[Set[int]]:
    if not arg:
        return None
    ids: Set[int] = set()
    for part in arg.split(','):
        part = part.strip()
        if part == '':
            continue
        try:
            ids.add(int(part))
        except ValueError:
            # 支持名字 -> id 转换
            if part in COCO_CLASSES:
                ids.add(COCO_CLASSES.index(part))
    return ids


def find_image_for_stem(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: any ext
    for p in images_dir.glob(f"{stem}.*"):
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
            return p
    return None


def collect_images_by_label(images_dir: Path, labels_dir: Path, class_filter: Optional[Set[int]]) -> Dict[int, List[Path]]:
    by_label: Dict[int, List[Path]] = {}
    for label_path in labels_dir.glob('*.txt'):
        stem = label_path.stem
        img_path = find_image_for_stem(images_dir, stem)
        if img_path is None:
            continue
        try:
            with label_path.open('r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        except Exception:
            continue
        present_ids: Set[int] = set()
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if class_filter is not None and cid not in class_filter:
                continue
            present_ids.add(cid)
        for cid in present_ids:
            by_label.setdefault(cid, []).append(img_path)
    return by_label


def main():
    parser = argparse.ArgumentParser(description="从datasets_trash抽样并合成YOLO图（按标签或来源）")
    parser.add_argument("--trash-dir", type=str, default=str(PROJECT_ROOT / "data" / "datasets_trash"), help="datasets_trash 目录")
    parser.add_argument("--mode", type=str, default="label", choices=["label", "source"], help="抽样模式: label(每类K张) 或 source(每来源K张)")
    parser.add_argument("--per-class", type=int, default=10, help="每个标签抽样数量（mode=label 有效）")
    parser.add_argument("--per-source", type=int, default=20, help="每个来源抽样数量（mode=source 有效）")
    parser.add_argument("--class-ids", type=str, default="", help="限制抽样的类别, 逗号分隔(支持id或名字)，为空表示全部")
    parser.add_argument("--output-subdir", type=str, default="trashimg", help="输出子目录名（位于trash根下）")
    args = parser.parse_args()

    trash_dir = Path(args.trash_dir)
    images_dir = trash_dir / "images"
    labels_dir = trash_dir / "labels"
    output_root = trash_dir / args.output_subdir

    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ 未找到必要目录: {images_dir} 或 {labels_dir}")
        return

    total_written = 0

    if args.mode == "source":
        by_source = collect_images_by_source(images_dir)
        if not by_source:
            print("❌ 未找到图片")
            return
        for source, paths in by_source.items():
            sample_paths = paths if len(paths) <= args.per_source else random.sample(paths, args.per_source)
            print(f"来源 {source}: 共 {len(paths)}，抽样 {len(sample_paths)}")
            for img_path in sample_paths:
                label_path = labels_dir / f"{img_path.stem}.txt"
                out_dir = output_root / source
                out_path = out_dir / img_path.stem
                ok = draw_yolo_boxes(img_path, label_path, out_path, COCO_CLASSES)
                if ok:
                    total_written += 1
    else:
        # 按标签抽样
        class_filter = parse_class_ids(args.class_ids)
        by_label = collect_images_by_label(images_dir, labels_dir, class_filter)
        if not by_label:
            print("❌ 未找到可用标签/图片")
            return
        # 稳定顺序：按类id排序
        for cid in sorted(by_label.keys()):
            cname = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"cls_{cid}"
            paths = by_label[cid]
            sample_paths = paths if len(paths) <= args.per_class else random.sample(paths, args.per_class)
            print(f"标签 {cid}({cname}): 共 {len(paths)}，抽样 {len(sample_paths)}")
            for img_path in sample_paths:
                label_path = labels_dir / f"{img_path.stem}.txt"
                out_dir = output_root / cname
                out_path = out_dir / img_path.stem
                ok = draw_yolo_boxes(img_path, label_path, out_path, COCO_CLASSES)
                if ok:
                    total_written += 1

    print(f"✅ 完成，输出 {total_written} 张到 {output_root}")


if __name__ == "__main__":
    main() 