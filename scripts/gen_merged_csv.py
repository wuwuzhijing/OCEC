import os
import csv
import argparse

# --- 配置参数 ---
BASE_DIR = '/ssddisk/guochuang/ocec/hq/'
# CSV 文件输出目录
OUTPUT_DIR = '/ssddisk/guochuang/ocec/list_hq_v3'

def generate_csv_list(merged_indices):
    """
    根据给定的索引列表 (merged_indices)，生成对应的 CSV 文件。
    文件路径为绝对路径。
    
    Args:
        merged_indices (list): 包含要处理的 merged_x 文件夹数字索引的列表。
    """
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✅ 创建输出目录: {OUTPUT_DIR}")

    for index in merged_indices:
        folder_name = f"merged_{index}"
        input_path = os.path.join(BASE_DIR, folder_name)
        output_filename = f"cropped_{folder_name}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        if not os.path.isdir(input_path):
            print(f"⚠️ 文件夹不存在，跳过: {input_path}")
            continue

        data_rows = []
        
        # 遍历 merged_x 文件夹下的所有子文件夹 (即 0 和 1)
        for class_label in ['0', '1']:
            class_path = os.path.join(input_path, class_label)
            
            if not os.path.isdir(class_path):
                print(f"⚠️ 类别子文件夹不存在，跳过: {class_path}")
                continue
            
            # 遍历子文件夹中的所有文件
            for filename in os.listdir(class_path):
                # 跳过隐藏文件
                if filename.startswith('.'):
                    continue
                
                # --- 关键修改：构建绝对路径 ---
                # 绝对路径 = BASE_DIR / merged_x / 类别 / 文件名
                absolute_file_path = os.path.join(input_path, class_label, filename)
                
                # 每一行的格式为：绝对路径，标签
                data_rows.append([absolute_file_path, class_label])
        
        # 将数据写入 CSV 文件
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入标题行
            writer.writerow(['File_Path', 'Label'])
            # 写入数据
            writer.writerows(data_rows)
            
        print(f"✅ 成功生成文件列表: {output_path} (共 {len(data_rows)} 行数据)")

if __name__ == "__main__":
    # --- 定制化部分 (Customization) ---
    
    # 示例：仅处理 merged_0, merged_1 和 merged_10
    indices_to_process = [0, 1, 2, 3, 4] 
    
    # 如果要处理所有 0 到 40 的文件夹，可以使用：
    # indices_to_process = list(range(41))
    
    # -----------------------------------
    
    generate_csv_list(indices_to_process)