#!/usr/bin/env python3
"""
按“标注数量”将 datasets 的 train/val 重新划分为 0.7/0.3。
- 基于 YOLO 标签文件统计每张图片的各类标注数
- 目标：对每个类别 class，总标注数的 70% 在 train，30% 在 val（允许少量误差）
- 启发式贪心移动图片，尽量让所有类别同时接近目标
- 同步移动 images/<split>/<stem>.* 与 labels/<split>/<stem>.txt

用法示例：
  python tests/rebalance_train_val_by_labels.py --datasets-root datasets --ratio 0.7 --dry-run --tolerance 3

参数：
- --datasets-root: 数据集根目录（含 images/, labels/）
- --train-split, --val-split: 训练/验证子目录名（默认 train/val）
- --ratio: 训练集比例（默认 0.7）
- --tolerance: 每类允许与目标的偏差阈值（默认 2 个标注）
- --max-moves: 最大移动图片数（默认 10000）
- --overwrite: 允许覆盖（默认否，自动添加后缀）
- --dry-run: 仅打印动作与统计，不落盘
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import shutil
from tqdm import tqdm
import math

SUPPORTED_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def read_label_counts(label_path: Path) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    try:
        with label_path.open('r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                counts[cid] = counts.get(cid, 0) + 1
    except Exception:
        pass
    return counts


def find_image_for_stem(images_dir: Path, stem: str) -> Optional[Path]:
    # 仅在指定 split 的 images 目录下寻找
    for ext in SUPPORTED_IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # 兜底：同目录任意后缀
    for p in images_dir.glob(f"{stem}.*"):
        if p.suffix.lower() in SUPPORTED_IMG_EXTS:
            return p
    return None


@dataclass
class ImageEntry:
    stem: str
    label_path: Path
    image_path: Optional[Path]
    class_counts: Dict[int, int]


def load_split_entries(labels_dir: Path, images_dir: Path) -> List[ImageEntry]:
    entries: List[ImageEntry] = []
    label_files = list(labels_dir.glob('*.txt'))
    for lp in tqdm(label_files, desc=f"扫描 {labels_dir.name} 标签", unit="file"):
        stem = lp.stem
        ip = find_image_for_stem(images_dir, stem)
        counts = read_label_counts(lp)
        entries.append(ImageEntry(stem=stem, label_path=lp, image_path=ip, class_counts=counts))
    return entries


def sum_class_counts(entries: List[ImageEntry]) -> Dict[int, int]:
    totals: Dict[int, int] = {}
    for e in entries:
        for cid, n in e.class_counts.items():
            totals[cid] = totals.get(cid, 0) + n
    return totals


def compute_targets(all_totals: Dict[int, int], ratio: float) -> Dict[int, int]:
    targets: Dict[int, int] = {}
    for cid, tot in all_totals.items():
        targets[cid] = int(round(tot * ratio))
    return targets


def current_counts(entries: List[ImageEntry]) -> Dict[int, int]:
    return sum_class_counts(entries)


def vector_add(base: Dict[int, int], delta: Dict[int, int], sign: int) -> None:
    for cid, n in delta.items():
        base[cid] = base.get(cid, 0) + sign * n


def score_move_to_train(deficits: Dict[int, int], entry: ImageEntry) -> int:
    score = 0
    for cid, need in deficits.items():
        if need <= 0:
            continue
        if cid in entry.class_counts:
            score += min(need, entry.class_counts[cid])
    return score


def score_move_to_val(surplus: Dict[int, int], entry: ImageEntry) -> int:
    score = 0
    for cid, over in surplus.items():
        if over <= 0:
            continue
        if cid in entry.class_counts:
            score += min(over, entry.class_counts[cid])
    return score


def ensure_unique_path(target: Path) -> Path:
    if not target.exists():
        return target
    parent, stem, ext = target.parent, target.stem, target.suffix
    idx = 1
    while True:
        cand = parent / f"{stem}_rebalance_{idx}{ext}"
        if not cand.exists():
            return cand
        idx += 1


def move_entry(entry: ImageEntry, src_imgs: Path, src_lbls: Path, dst_imgs: Path, dst_lbls: Path, dry_run: bool, overwrite: bool) -> bool:
    # 源路径
    src_label = entry.label_path
    src_image = entry.image_path
    if not src_label.exists():
        return False
    if src_image is None or not src_image.exists():
        # 允许没有图片时只移动标签？为保持一致性，这里跳过该条
        return False

    dst_label = dst_lbls / src_label.name
    dst_image = dst_imgs / src_image.name
    if not overwrite:
        dst_label = ensure_unique_path(dst_label)
        dst_image = ensure_unique_path(dst_image)

    if not dry_run:
        dst_lbls.mkdir(parents=True, exist_ok=True)
        dst_imgs.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_label), str(dst_label))
        shutil.move(str(src_image), str(dst_image))
    return True


def within_tolerance(cur: Dict[int, int], tgt: Dict[int, int], tol: int) -> bool:
    for cid, t in tgt.items():
        if abs(cur.get(cid, 0) - t) > tol:
            return False
    return True


def rebalance(dataset_root: Path, train_split: str, val_split: str, ratio: float, tolerance: int, max_moves: int, dry_run: bool, overwrite: bool) -> None:
    train_labels = dataset_root / 'labels' / train_split
    val_labels = dataset_root / 'labels' / val_split
    train_images = dataset_root / 'images' / train_split
    val_images = dataset_root / 'images' / val_split

    if not train_labels.exists() or not val_labels.exists():
        print(f"❌ 标签目录不存在: {train_labels} 或 {val_labels}")
        return

    train_entries = load_split_entries(train_labels, train_images)
    val_entries = load_split_entries(val_labels, val_images)

    all_totals = sum_class_counts(train_entries)  # start with train
    vector_add(all_totals, sum_class_counts(val_entries), +1)  # add val

    targets = compute_targets(all_totals, ratio)
    cur_train = current_counts(train_entries)

    print("目标(训练集)标注数：前若干类")
    for cid in sorted(targets.keys())[:10]:
        print(f"  class {cid}: target={targets[cid]}, current={cur_train.get(cid, 0)}")

    # 用集合跟踪尚未移动的候选
    movable_from_val: Set[str] = set(e.stem for e in val_entries)
    movable_from_train: Set[str] = set(e.stem for e in train_entries)

    moves_done = 0
    improved = True

    while moves_done < max_moves and not within_tolerance(cur_train, targets, tolerance) and improved:
        improved = False

        # 计算当前偏差
        deficits = {cid: max(0, targets[cid] - cur_train.get(cid, 0)) for cid in targets}
        surplus = {cid: max(0, cur_train.get(cid, 0) - targets[cid]) for cid in targets}

        # 从 val -> train：填补缺口
        best_v: Optional[Tuple[int, ImageEntry]] = None
        for e in val_entries:
            if e.stem not in movable_from_val:
                continue
            sc = score_move_to_train(deficits, e)
            if sc > 0 and (best_v is None or sc > best_v[0]):
                best_v = (sc, e)
        if best_v is not None:
            _, e = best_v
            ok = move_entry(e, val_images, val_labels, train_images, train_labels, dry_run, overwrite)
            if ok:
                # 更新集合与计数
                movable_from_val.remove(e.stem)
                movable_from_train.add(e.stem)
                train_entries.append(e)
                val_entries.remove(e)
                vector_add(cur_train, e.class_counts, +1)
                moves_done += 1
                improved = True
                continue  # 下一轮重算偏差

        # 从 train -> val：削减过量
        best_t: Optional[Tuple[int, ImageEntry]] = None
        for e in train_entries:
            if e.stem not in movable_from_train:
                continue
            sc = score_move_to_val(surplus, e)
            if sc > 0 and (best_t is None or sc > best_t[0]):
                best_t = (sc, e)
        if best_t is not None:
            _, e = best_t
            ok = move_entry(e, train_images, train_labels, val_images, val_labels, dry_run, overwrite)
            if ok:
                movable_from_train.remove(e.stem)
                movable_from_val.add(e.stem)
                val_entries.append(e)
                train_entries.remove(e)
                vector_add(cur_train, e.class_counts, -1)
                moves_done += 1
                improved = True
                continue

    # 总结
    print(f"完成。移动次数: {moves_done} (max {max_moves})")
    print("最终偏差（train - target）：")
    for cid in sorted(targets.keys()):
        diff = cur_train.get(cid, 0) - targets[cid]
        if diff != 0:
            print(f"  class {cid}: diff {diff:+d}")


def main():
    ap = argparse.ArgumentParser(description='按标注数将 train/val 重新划分为 0.7/0.3')
    ap.add_argument('--datasets-root', type=str, default='datasets', help='数据集根目录')
    ap.add_argument('--train-split', type=str, default='train', help='训练集目录名')
    ap.add_argument('--val-split', type=str, default='val', help='验证集目录名')
    ap.add_argument('--ratio', type=float, default=0.7, help='训练集比例，默认0.7')
    ap.add_argument('--tolerance', type=int, default=2, help='每类允许与目标的偏差')
    ap.add_argument('--max-moves', type=int, default=10000, help='最多移动图片数')
    ap.add_argument('--overwrite', action='store_true', help='允许覆盖同名文件（默认添加后缀）')
    ap.add_argument('--dry-run', action='store_true', help='仅打印动作与统计，不实际移动')
    args = ap.parse_args()

    rebalance(
        dataset_root=Path(args.datasets_root),
        train_split=args.train_split,
        val_split=args.val_split,
        ratio=args.ratio,
        tolerance=args.tolerance,
        max_moves=args.max_moves,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main() 