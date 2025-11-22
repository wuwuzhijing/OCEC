import cv2
import numpy as np
import os
import random
from glob import glob
from typing import List, Tuple

def create_raw_data_grid_flexible(image_dir: str, grid_cols: int = 10, grid_rows: int = 10, output_filename: str = "QA_Raw_Flexible_Grid.png"):
    
    file_patterns = [os.path.join(image_dir, f'*.{ext}') for ext in ['jpg', 'png', 'jpeg']]
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob(pattern))

    if not all_files:
        print(f"错误：文件夹 {image_dir} 中未找到任何图片。")
        return

    # 1. 扫描样本，确定最大尺寸 (W_max, H_max)
    max_w, max_h = 0, 0
    # 随机选择足够的样本进行尺寸分析
    sample_files = random.sample(all_files, min(len(all_files), grid_cols * grid_rows * 2)) 
    
    for file_path in sample_files:
        # 不指定任何flag，按原始通道读取
        img = cv2.imread(file_path) 
        if img is not None:
            # 灰度图的 shape 是 (H, W)，彩色图是 (H, W, C)
            h, w = img.shape[:2]
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            
    if max_w == 0 or max_h == 0:
        print("错误：未能检测到有效图片尺寸。")
        return

    # 确定最终要处理的样本数量
    total_needed = grid_cols * grid_rows
    selected_files = random.sample(all_files, min(len(all_files), total_needed))

    # 2. 逐个处理、填充和堆叠
    grid_images: List[np.ndarray] = []
    current_row: List[np.ndarray] = []
    
    # 统一填充值为白色（BGR三通道）
    PADDING_COLOR = [255, 255, 255] 
    
    for i, file_path in enumerate(selected_files):
        # 再次按原始通道读取
        img = cv2.imread(file_path) 
        
        if img is None:
            continue
        
        # 统一为三通道 BGR 格式
        if len(img.shape) == 2:
            # 单通道灰度图，复制通道转换为 BGR，内容不变
            img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            # 彩色图，直接使用
            img_3ch = img
            
        h, w = img_3ch.shape[:2]
        
        # 计算填充量 (Padding)
        pad_h, pad_w = max_h - h, max_w - w
        
        pad_top, pad_left = pad_h // 2, pad_w // 2
        pad_bottom, pad_right = pad_h - pad_top, pad_w - pad_left
        
        # 3. 使用 cv2.copyMakeBorder 添加白色填充
        # value 必须是三元组
        padded_img = cv2.copyMakeBorder(
            img_3ch, 
            pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, 
            value=PADDING_COLOR
        )

        current_row.append(padded_img)
        
        # 水平堆叠
        if len(current_row) == grid_cols:
            row_image = np.hstack(current_row)
            grid_images.append(row_image)
            current_row = [] 

    # 5. 垂直堆叠
    if grid_images:
        final_grid = np.vstack(grid_images)
        
        # 保存为 PNG 格式（无损）
        cv2.imwrite(output_filename, final_grid)
        print(f"成功创建原始通道网格图：{output_filename}")
    else:
        print("未能创建任何图片网格。")

# --- 使用示例 ---
# 目标：从指定文件夹中抽取 10x10 的网格图
create_raw_data_grid_flexible(
    image_dir='cropped/100000011/', # 请替换为您的文件夹路径
    grid_cols=10, 
    grid_rows=10, 
    output_filename='QA_Raw_Data_Flexible_Grid_100000011.png'
)