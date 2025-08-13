#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签分布分析脚本
用于统计训练集和验证集中各类别的样本分布情况
重点分析person和car类别的样本数量
"""

import os
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# obstacle.yaml中的类别映射
class_names = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'traffic_cone',
    4: 'barrier',
    5: 'curb',
    6: 'manhole',
    7: 'stairs',
    8: 'railing',
    9: 'pole',
    10: 'water',
    11: 'sand',
    12: 'snow',
    13: 'tripod',
    14: 'barrel',
    15: 'motorcycle',
    16: 'bus',
    17: 'train',
    18: 'truck',
    19: 'traffic_light',
    20: 'fire_hydrant',
    21: 'stop_sign',
    22: 'bench',
    23: 'chair',
    24: 'red_light',
    25: 'yellow_light',
    26: 'green_light',
    27: 'off_light',
    28: 'crosswalk',
    29: 'guide_arrows'
}

def analyze_labels(labels_dir):
    """分析标签目录中的类别分布"""
    print(f"分析标签目录: {labels_dir}")
    
    # 统计每个类别的样本数
    class_counts = defaultdict(int)
    total_annotations = 0
    total_files = 0
    
    # 获取所有txt文件
    label_files = list(Path(labels_dir).glob("*.txt"))
    
    if not label_files:
        print(f"❌ 未找到标签文件在 {labels_dir}")
        return class_counts, total_annotations, total_files
    
    print(f"找到 {len(label_files)} 个标签文件")
    
    for label_file in tqdm(label_files, desc="处理标签文件"):
        total_files += 1
        
        if label_file.stat().st_size == 0:
            # 空文件表示无标注对象
            continue
            
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # 解析YOLO格式: class_id x_center y_center width height
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_annotations += 1
    
    return class_counts, total_annotations, total_files

def print_distribution_report(train_counts, val_counts, train_total, val_total, train_files, val_files):
    """打印分布报告"""
    print("\n" + "="*60)
    print("标签分布分析报告")
    print("="*60)
    
    print(f"\n📊 总体统计:")
    print(f"训练集文件数: {train_files}")
    print(f"验证集文件数: {val_files}")
    print(f"训练集标注数: {train_total}")
    print(f"验证集标注数: {val_total}")
    
    # 合并统计
    all_classes = set(train_counts.keys()) | set(val_counts.keys())
    
    print(f"\n📋 各类别分布:")
    print(f"{'类别ID':<8} {'类别名称':<15} {'训练集':<8} {'验证集':<8} {'总计':<8} {'百分比':<8}")
    print("-" * 70)
    
    # 按总数排序
    class_totals = []
    for class_id in all_classes:
        train_count = train_counts.get(class_id, 0)
        val_count = val_counts.get(class_id, 0)
        total_count = train_count + val_count
        class_totals.append((class_id, total_count, train_count, val_count))
    
    class_totals.sort(key=lambda x: x[1], reverse=True)
    
    grand_total = train_total + val_total
    
    for class_id, total_count, train_count, val_count in class_totals:
        class_name = class_names.get(class_id, f"unknown_{class_id}")
        percentage = (total_count / grand_total) * 100 if grand_total > 0 else 0
        
        print(f"{class_id:<8} {class_name:<15} {train_count:<8} {val_count:<8} {total_count:<8} {percentage:.2f}%")
    
    # 重点关注person和car
    print(f"\n🔍 关键类别分析:")
    person_train = train_counts.get(0, 0)
    person_val = val_counts.get(0, 0)
    person_total = person_train + person_val
    
    car_train = train_counts.get(2, 0)
    car_val = val_counts.get(2, 0)
    car_total = car_train + car_val
    
    print(f"Person (类别0):")
    print(f"  - 训练集: {person_train} 个样本")
    print(f"  - 验证集: {person_val} 个样本")
    print(f"  - 总计: {person_total} 个样本 ({(person_total/grand_total)*100:.2f}%)")
    
    print(f"\nCar (类别2):")
    print(f"  - 训练集: {car_train} 个样本")
    print(f"  - 验证集: {car_val} 个样本")
    print(f"  - 总计: {car_total} 个样本 ({(car_total/grand_total)*100:.2f}%)")
    
    # 问题诊断
    print(f"\n🔧 问题诊断:")
    if person_total == 0:
        print("❌ Person类别样本数为0！这是识别不到人的主要原因。")
    elif person_total < 100:
        print(f"⚠️  Person类别样本数过少({person_total})，可能导致识别效果差。")
    
    if car_total == 0:
        print("❌ Car类别样本数为0！这是识别不到汽车的主要原因。")
    elif car_total < 100:
        print(f"⚠️  Car类别样本数过少({car_total})，可能导致识别效果差。")

def main():
    """主函数"""
    print("BlindStar 标签分布分析")
    print("=" * 30)
    
    # 数据集路径
    dataset_root = Path("e:/BlindStar/datasets")
    train_labels_dir = dataset_root / "labels" / "train"
    val_labels_dir = dataset_root / "labels" / "val"
    
    # 检查路径存在性
    if not train_labels_dir.exists():
        print(f"❌ 训练标签目录不存在: {train_labels_dir}")
        return
    
    if not val_labels_dir.exists():
        print(f"❌ 验证标签目录不存在: {val_labels_dir}")
        return
    
    # 分析训练集
    print("\n📁 分析训练集...")
    train_counts, train_total, train_files = analyze_labels(train_labels_dir)
    
    # 分析验证集
    print("\n📁 分析验证集...")
    val_counts, val_total, val_files = analyze_labels(val_labels_dir)
    
    # 生成报告
    print_distribution_report(train_counts, val_counts, train_total, val_total, train_files, val_files)

if __name__ == "__main__":
    main()
