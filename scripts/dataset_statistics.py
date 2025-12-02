import csv
import os
from collections import defaultdict
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt

# --- 配置参数 ---
CSV_DIR = '/ssddisk/guochuang/ocec/list_hq'
REPORT_DIR = '/103/guochuang/Code/myOCEC/logs/dataset/list_hq/'
REPORT_FILENAME = 'dataset_stats_report.txt'
REPORT_PATH = os.path.join(REPORT_DIR, REPORT_FILENAME)
PIXEL_SAMPLE_LIMIT = 5000000  # 限制像素采样数量，避免内存溢出

# 想要统计的 CSV 文件列表（请根据需要修改）
CSV_FILENAMES = [
    f"annotation_{i:04d}.csv" for i in range(24, 25)] + [
    f"cropped_merged_{i:01d}.csv" for i in range(0, 5)
]

# ===============================================
# A. 绘图函数
# ===============================================

def plot_with_stats(data, title, xlabel, path):
    """绘制带统计标记的直方图"""
    if not data:
        return
    
    data = np.array(data)
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)

    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 标记均值 (Mean)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    
    # 标记中位数 (Median)
    plt.axvline(median_val, color='green', linestyle='solid', linewidth=2, label=f'Median: {median_val:.2f}')
    
    # 添加标准差文字
    plt.text(0.95, 0.95, f'Std Dev: {std_val:.2f}', transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6))
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    plt.savefig(path)
    plt.close()
    print(f"✅ {title} 图已保存: {path}")
    return path

def plot_class_distribution(class_counts, path):
    """绘制类别数量柱状图"""
    if not class_counts:
        return
        
    labels = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    plt.title('Global Class Distribution')
    plt.xlabel('Class Label')
    plt.ylabel('Sample Count')
    
    # 在柱子上显示数量
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + len(counts)*100, f'{yval:,}', ha='center', va='bottom', fontsize=9)
        
    plt.grid(axis='y', alpha=0.5)
    
    plt.savefig(path)
    plt.close()
    print(f"✅ 类别分布图已保存: {path}")
    return path

def plot_pixel_distribution(raw_pixels_sample, path):
    """绘制像素值分布直方图 (R/G/B 三通道)"""
    if len(raw_pixels_sample) == 0:
        return
        
    # 将采样的像素数据转为 Numpy 数组 (Shape: N x 3)
    pixels_array = np.array(raw_pixels_sample)
    
    plt.figure(figsize=(12, 7))
    colors = ['red', 'green', 'blue']
    labels = ['Red Channel', 'Green Channel', 'Blue Channel']
    
    # 绘制三通道的直方图
    for i in range(3):
        # bins=50, 范围[0, 1] 因为数据已经被归一化到 [0, 1]
        plt.hist(pixels_array[:, i], bins=50, range=[0, 1], alpha=0.6, color=colors[i], label=labels[i], edgecolor='none')
        
    plt.title('Pixel Value Distribution (Normalized [0, 1])')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    plt.savefig(path)
    plt.close()
    print(f"✅ 像素分布图已保存: {path}")
    return path


# ===============================================
# B. 分析函数
# ===============================================

def analyze_dataset(csv_filenames, check_pixel_stats=False, generate_plots=False):
    # ... (初始化和日志函数与 V4.0 相同) ...
    total_samples = 0
    total_missing_files = 0
    global_class_counts = defaultdict(int)
    
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    total_pixels = 0
    
    raw_heights = []
    raw_widths = []
    # 新增：用于像素分布采样的列表
    raw_pixels_sample = [] 
    
    file_analysis_details = {}
    report_buffer = []

    def log_report(message):
        report_buffer.append(message)
        print(message)
    
    log_report(f"--- 开始分析 {len(csv_filenames)} 个 CSV 文件 ---")
    
    # ... (文件遍历和数据收集循环) ...
    for filename in csv_filenames:
        csv_path = os.path.join(CSV_DIR, filename)
        if not os.path.exists(csv_path):
            log_report(f"⚠️ 警告：文件未找到，跳过: {csv_path}")
            continue

        log_report(f"\n> 正在处理文件: {filename}")
        
        file_samples = 0
        file_class_counts = defaultdict(int)

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                next(reader) 
            except StopIteration:
                continue

            for row in reader:
                # ... (行处理和计数逻辑省略) ...
                if len(row) != 2:
                    continue
                
                file_path, label = row
                
                total_samples += 1
                global_class_counts[label] += 1
                file_samples += 1
                file_class_counts[label] += 1
                
                if not os.path.exists(file_path):
                    total_missing_files += 1
                    continue
                    
                # 图像属性统计 (需要读取图像)
                if check_pixel_stats or generate_plots:
                    try:
                        img = Image.open(file_path).convert('RGB')
                        width, height = img.size
                        img_array = np.array(img, dtype=np.float32) / 255.0
                        
                        # I. 像素统计 (Mean/Std)
                        if check_pixel_stats:
                            total_pixels += height * width
                            pixel_sum += np.sum(img_array, axis=(0, 1))
                            pixel_sq_sum += np.sum(img_array**2, axis=(0, 1))

                        # II. 绘图数据收集
                        if generate_plots:
                            raw_heights.append(height)
                            raw_widths.append(width)
                            
                            # **像素采样逻辑**：只采样不超过 PIXEL_SAMPLE_LIMIT 的数量
                            if len(raw_pixels_sample) < PIXEL_SAMPLE_LIMIT:
                                # 随机选择当前图像中的像素，按比例采样
                                flat_pixels = img_array.reshape(-1, 3)
                                sample_size = min(len(flat_pixels), int(PIXEL_SAMPLE_LIMIT / total_samples))
                                
                                # 随机选择索引
                                indices = np.random.choice(flat_pixels.shape[0], size=sample_size, replace=False)
                                raw_pixels_sample.extend(flat_pixels[indices])
                            
                    except Exception as e:
                        total_missing_files += 1

        if file_samples > 0:
            file_analysis_details[filename] = {'samples': file_samples, 'class_counts': file_class_counts}

    # ... (结果计算和写入缓冲区逻辑省略，与 V4.0 相同) ...
    global_mean = pixel_sum / total_pixels if total_pixels > 0 else np.zeros(3)
    global_std = np.sqrt(pixel_sq_sum / total_pixels - global_mean**2) if total_pixels > 0 else np.zeros(3)
    
    # 报告主体内容写入 report_buffer 
    log_report("\n" + "="*70)
    log_report("                 数据集综合统计报告")
    log_report("="*70)
    log_report(f"总样本数 (Total Samples): {total_samples:,}")
    log_report(f"文件缺失/无效数 (Missing/Invalid Files): {total_missing_files:,}")
    # ... (A. 文件级类别分布) ...
    log_report("\n--- A. 文件级类别分布 (Per-File Class Balance) ---")
    for filename, details in file_analysis_details.items():
        counts = details['class_counts']
        output = f" {filename} ({details['samples']:,} rows): "
        if '0' in counts and '1' in counts and counts['0'] > 0 and counts['1'] > 0:
            ratio = counts['0'] / (counts['0'] + counts['1']) * 100
            output += f" 0: {counts['0']:,}, 1: {counts['1']:,} (0类占比: {ratio:.2f}%)"
        else:
            output += f" 分布不完整或单类别: {dict(counts)}"
        log_report(output)
    
    # ... (B. 全局像素统计) ...
    if check_pixel_stats:
        log_report("\n--- B. 全局像素统计 (Normalization Parameters) ---")
        log_report(f"图像通道数: {3} (默认为 RGB)")
        log_report(f"全局均值 (Mean, R/G/B): {global_mean}")
        log_report(f"全局标准差 (Std Dev, R/G/B): {global_std}")
        log_report("\n💡 建议：将这些值用于您的 DataLoader/Transforms 配置中。")
    else:
        log_report("\n💡 像素统计未运行。")

    # ... (C. 总体类别分布) ...
    log_report("\n--- C. 总体类别分布 (Global Class Distribution) ---")
    sorted_counts = sorted(global_class_counts.items(), key=lambda item: item[1], reverse=True)
    if total_samples > 0:
        for label, count in sorted_counts:
            percentage = (count / total_samples) * 100
            log_report(f"类别 {label}: {count:,} ({percentage:.2f}%)")
    log_report("="*70)
    
    # --- 最终写入文件 ---
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
        
    final_report_content = []
    # ... (写入 CSV 列表和报告主体逻辑省略，与 V4.0 相同) ...
    final_report_content.append("="*70)
    final_report_content.append(f"报告生成时间: {os.popen('date').read().strip()}")
    final_report_content.append(f"报告文件路径: {REPORT_PATH}")
    final_report_content.append("\n--- 分析的 CSV 文件列表 ---")
    for csv_file in csv_filenames:
        final_report_content.append(f"- {csv_file}")
    final_report_content.append("--- 报告主体 ---")
    final_report_content.extend(report_buffer)

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_report_content))
        
    print(f"\n✅ 统计数据已写入报告: {REPORT_PATH}")
    
    # 绘图返回
    if generate_plots:
        return raw_heights, raw_widths, global_class_counts, raw_pixels_sample
    else:
        return None, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集 CSV 文件统计工具 (V5.0 - 全面可视化)")
    parser.add_argument(
        '--check_pixel_stats', 
        action='store_true', 
        help="启用耗时的像素均值和标准差计算。"
    )
    parser.add_argument(
        '--generate_plots', 
        action='store_true', 
        help="生成图像尺寸、类别和像素分布图表（需要读取图像文件）。"
    )
    args = parser.parse_args()
    
    read_images = args.check_pixel_stats or args.generate_plots
    
    heights, widths, class_counts, pixels = analyze_dataset(CSV_FILENAMES, read_images, args.generate_plots)
    
    if args.generate_plots:
        plot_results = []
        plot_results.append(plot_with_stats(heights, 'Distribution of Image Heights (with Stats)', 'Height (Pixels)', os.path.join(REPORT_DIR, 'height_stats_histogram.png')))
        plot_results.append(plot_with_stats(widths, 'Distribution of Image Widths (with Stats)', 'Width (Pixels)', os.path.join(REPORT_DIR, 'width_stats_histogram.png')))
        plot_results.append(plot_class_distribution(class_counts, os.path.join(REPORT_DIR, 'class_distribution_bar.png')))
        plot_results.append(plot_pixel_distribution(pixels, os.path.join(REPORT_DIR, 'pixel_distribution_histogram.png')))
        
        print("\n--- 所有图表生成完成 ---")
        for path in plot_results:
            if path:
                print(f"🖼️ {os.path.basename(path)}")