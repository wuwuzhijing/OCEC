#!/usr/bin/env python3
"""分析类间分离指标的变化趋势"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tensorboard.backend.event_processing import event_accumulator
    import glob
    import matplotlib.pyplot as plt
    import numpy as np
    
    log_dir = 'runs/ocec_hq'
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    
    if not event_files:
        print("未找到 TensorBoard 事件文件")
        sys.exit(1)
    
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # 提取类间分离指标
    metrics = {
        'train/intra': [],
        'train/inter': [],
        'train/fisher': [],
        'train/bhattacharyya': [],
        'val/intra': [],
        'val/inter': [],
        'val/fisher': [],
        'val/bhattacharyya': []
    }
    
    for metric_name in metrics.keys():
        if metric_name in ea.scalars.Keys():
            scalar_events = ea.scalars.Items(metric_name)
            for event in scalar_events:
                if event.step <= 60:
                    metrics[metric_name].append((int(event.step), event.value))
            metrics[metric_name].sort(key=lambda x: x[0])
    
    # 打印详细表格
    print("\n" + "="*120)
    print("类间分离指标详细数据 (Epoch 10-60)")
    print("="*120)
    print(f"{'Epoch':<8} | {'Train Intra':<12} | {'Train Inter':<12} | {'Train Fisher':<14} | {'Train Bhatt':<13} | {'Val Intra':<11} | {'Val Inter':<11} | {'Val Fisher':<13} | {'Val Bhatt':<12}")
    print("-"*120)
    
    epochs = sorted(set([e for metric_data in metrics.values() for e, _ in metric_data]))
    for epoch in epochs:
        if epoch % 10 == 0:
            row = [f"{epoch:<8}"]
            for metric_name in ['train/intra', 'train/inter', 'train/fisher', 'train/bhattacharyya', 
                               'val/intra', 'val/inter', 'val/fisher', 'val/bhattacharyya']:
                value = next((v for e, v in metrics[metric_name] if e == epoch), None)
                if value is not None:
                    row.append(f"{value:<12.4f}" if 'Fisher' not in metric_name else f"{value:<14.4f}")
                else:
                    row.append(f"{'N/A':<12}")
            print(" | ".join(row))
    
    # 趋势分析
    print("\n" + "="*120)
    print("趋势分析 (Epoch 10 → 60)")
    print("="*120)
    
    def analyze_trend(metric_name, label):
        if len(metrics[metric_name]) >= 2:
            first_epoch, first_val = metrics[metric_name][0]
            last_epoch, last_val = metrics[metric_name][-1]
            change = last_val - first_val
            change_pct = (change / abs(first_val) * 100) if abs(first_val) > 1e-8 else 0
            
            # 判断趋势
            if 'fisher' in metric_name:
                trend = "✅ 改善" if change > 0 else "❌ 恶化" if change < 0 else "➡️ 稳定"
                ideal = "↑ (越大越好)"
            elif 'inter' in metric_name:
                trend = "✅ 改善" if change > 0 else "⚠️ 减小" if change < 0 else "➡️ 稳定"
                ideal = "↑ (越大越好)"
            elif 'intra' in metric_name:
                trend = "✅ 改善" if change < 0 else "❌ 增大" if change > 0 else "➡️ 稳定"
                ideal = "↓ (越小越好)"
            else:
                trend = "➡️ 稳定"
                ideal = ""
            
            print(f"{label:25s}: {first_val:10.4f} → {last_val:10.4f} ({change:+10.4f}, {change_pct:+7.2f}%) {trend} {ideal}")
            return first_val, last_val, change, change_pct
    
    print("\n【关键指标】")
    analyze_trend('train/fisher', '训练集 Fisher Ratio')
    analyze_trend('val/fisher', '验证集 Fisher Ratio')
    print()
    analyze_trend('train/inter', '训练集 类间距离')
    analyze_trend('val/inter', '验证集 类间距离')
    print()
    analyze_trend('train/intra', '训练集 类内距离')
    analyze_trend('val/intra', '验证集 类内距离')
    
    # 计算 Fisher Ratio 的变化率
    print("\n" + "="*120)
    print("Fisher Ratio 变化分析 (类间分离度)")
    print("="*120)
    train_fisher_vals = [v for _, v in metrics['train/fisher']]
    val_fisher_vals = [v for _, v in metrics['val/fisher']]
    
    if len(train_fisher_vals) >= 2:
        train_improvement = (train_fisher_vals[-1] / train_fisher_vals[0] - 1) * 100
        print(f"训练集 Fisher Ratio: {train_fisher_vals[0]:.4f} → {train_fisher_vals[-1]:.4f}")
        print(f"  提升倍数: {train_fisher_vals[-1] / train_fisher_vals[0]:.2f}x")
        print(f"  提升百分比: {train_improvement:+.2f}%")
    
    if len(val_fisher_vals) >= 2:
        val_improvement = (val_fisher_vals[-1] / val_fisher_vals[0] - 1) * 100
        print(f"验证集 Fisher Ratio: {val_fisher_vals[0]:.4f} → {val_fisher_vals[-1]:.4f}")
        print(f"  提升倍数: {val_fisher_vals[-1] / val_fisher_vals[0]:.2f}x")
        print(f"  提升百分比: {val_improvement:+.2f}%")
    
    # 综合评估
    print("\n" + "="*120)
    print("综合评估")
    print("="*120)
    
    train_fisher_improved = len(train_fisher_vals) >= 2 and train_fisher_vals[-1] > train_fisher_vals[0]
    val_fisher_improved = len(val_fisher_vals) >= 2 and val_fisher_vals[-1] > val_fisher_vals[0]
    train_intra_decreased = len(metrics['train/intra']) >= 2 and metrics['train/intra'][-1][1] < metrics['train/intra'][0][1]
    val_intra_decreased = len(metrics['val/intra']) >= 2 and metrics['val/intra'][-1][1] < metrics['val/intra'][0][1]
    
    positive_signs = sum([train_fisher_improved, val_fisher_improved, train_intra_decreased, val_intra_decreased])
    
    if positive_signs >= 3:
        print("✅ 训练状态：良好")
        print("   - Fisher Ratio 显著提升，说明 CosFace 正在有效拉开类间距离")
        print("   - 类内距离减小，说明类内更紧凑")
        print("   - CosFace 的 margin 机制正在发挥作用")
    elif positive_signs >= 2:
        print("⚠️ 训练状态：一般")
        print("   - 部分指标改善，但仍有优化空间")
    else:
        print("❌ 训练状态：异常")
        print("   - 类间分离指标未按预期改善，需要检查训练配置")
    
    # 创建趋势图
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('类间分离指标变化趋势 (Epoch 10-60)', fontsize=16, fontweight='bold')
        
        # 1. Fisher Ratio
        ax1 = axes[0, 0]
        if metrics['train/fisher']:
            epochs_train, vals_train = zip(*metrics['train/fisher'])
            ax1.plot(epochs_train, vals_train, 'o-', label='Train', linewidth=2, markersize=6)
        if metrics['val/fisher']:
            epochs_val, vals_val = zip(*metrics['val/fisher'])
            ax1.plot(epochs_val, vals_val, 's-', label='Val', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Fisher Ratio', fontsize=11)
        ax1.set_title('Fisher Ratio (类间分离度，越大越好)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Inter-class Distance
        ax2 = axes[0, 1]
        if metrics['train/inter']:
            epochs_train, vals_train = zip(*metrics['train/inter'])
            ax2.plot(epochs_train, vals_train, 'o-', label='Train', linewidth=2, markersize=6)
        if metrics['val/inter']:
            epochs_val, vals_val = zip(*metrics['val/inter'])
            ax2.plot(epochs_val, vals_val, 's-', label='Val', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Inter-class Distance', fontsize=11)
        ax2.set_title('类间距离 (越大越好)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Intra-class Distance
        ax3 = axes[1, 0]
        if metrics['train/intra']:
            epochs_train, vals_train = zip(*metrics['train/intra'])
            ax3.plot(epochs_train, vals_train, 'o-', label='Train', linewidth=2, markersize=6)
        if metrics['val/intra']:
            epochs_val, vals_val = zip(*metrics['val/intra'])
            ax3.plot(epochs_val, vals_val, 's-', label='Val', linewidth=2, markersize=6)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Intra-class Distance', fontsize=11)
        ax3.set_title('类内距离 (越小越好)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Bhattacharyya Distance
        ax4 = axes[1, 1]
        if metrics['train/bhattacharyya']:
            epochs_train, vals_train = zip(*metrics['train/bhattacharyya'])
            ax4.plot(epochs_train, vals_train, 'o-', label='Train', linewidth=2, markersize=6)
        if metrics['val/bhattacharyya']:
            epochs_val, vals_val = zip(*metrics['val/bhattacharyya'])
            ax4.plot(epochs_val, vals_val, 's-', label='Val', linewidth=2, markersize=6)
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Bhattacharyya Distance', fontsize=11)
        ax4.set_title('Bhattacharyya 距离 (越大越好)', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        trend_plot_path = os.path.join(log_dir, 'class_separation_trend.png')
        plt.savefig(trend_plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 趋势图已保存: {trend_plot_path}")
        plt.close()
    except Exception as e:
        print(f"\n⚠️ 无法生成趋势图: {e}")
    
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装: pip install tensorboard matplotlib numpy")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

