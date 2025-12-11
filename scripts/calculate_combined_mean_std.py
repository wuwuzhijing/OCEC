#!/usr/bin/env python3
"""
计算两个数据集合并后的总体mean和std

公式：
mean_global = (NA * mean_A + NB * mean_B) / (NA + NB)
var_global = (NA * (std_A² + mean_A²) + NB * (std_B² + mean_B²)) / (NA + NB) - (mean_global)²
std_global = sqrt(var_global)
"""

import numpy as np

# ========== 数据集A（25W public + 5W private）==========
NA = 296735

mean_A = np.array([0.36043344,0.28117363,0.25993526])
std_A = np.array([0.20355277,0.18570374,0.1877159])

# ========== 数据集B（40W private）==========
NB = 470811
mean_B = np.array([0.46961902,0.46596417,0.46912243])
std_B = np.array([0.17677383,0.17855343,0.17680589])

print("=" * 60)
print("数据集统计信息")
print("=" * 60)
print(f"\n数据集A（25W public + 5W private）:")
print(f"  样本数: {NA:,}")
print(f"  Mean (R/G/B): {mean_A}")
print(f"  Std  (R/G/B): {std_A}")

print(f"\n数据集B（40W private）:")
print(f"  样本数: {NB:,}")
print(f"  Mean (R/G/B): {mean_B}")
print(f"  Std  (R/G/B): {std_B}")

# ========== 计算总体统计 ==========
print("\n" + "=" * 60)
print("计算总体统计（合并数据集A和B）")
print("=" * 60)

# 计算总体mean
mean_global = (NA * mean_A + NB * mean_B) / (NA + NB)
print(f"\n总体 Mean (R/G/B): {mean_global}")

# 计算总体variance
# var = E[X²] - E[X]²
# E[X²] = std² + mean²
var_global = (
    (NA * (std_A**2 + mean_A**2) + NB * (std_B**2 + mean_B**2)) / (NA + NB)
    - mean_global**2
)

# 计算总体std
std_global = np.sqrt(var_global)
print(f"总体 Std  (R/G/B): {std_global}")

# ========== 输出Python代码格式 ==========
print("\n" + "=" * 60)
print("Python代码格式（可直接复制使用）")
print("=" * 60)
print(f"\nDEFAULT_MEAN = {mean_global.tolist()}")
print(f"DEFAULT_STD  = {std_global.tolist()}")

# ========== 验证计算 ==========
print("\n" + "=" * 60)
print("验证计算")
print("=" * 60)
print(f"\n总样本数: {NA + NB:,}")
print(f"Mean验证: (296735 * {mean_A[0]:.6f} + 470811 * {mean_B[0]:.6f}) / {NA + NB} = {mean_global[0]:.6f}")

# 计算每个通道的详细值
print("\n详细计算（每个通道）:")
for i, channel in enumerate(['R', 'G', 'B']):
    mean_val = mean_global[i]
    std_val = std_global[i]
    print(f"\n{channel}通道:")
    print(f"  Mean: {mean_val:.8f}")
    print(f"  Std:  {std_val:.8f}")