# Validation Framework

验证框架用于解决审稿人关于油品类型识别模型过拟合的关切。

## 目录结构

```
src/validation/
├── __init__.py                   # Package initialization
├── validation.py                 # 核心验证框架（数据加载、模型创建）
├── leave_replicate_out.py        # 留一平行样本交叉验证
├── multi_seed_cv.py              # 多随机种子交叉验证（含置信区间）
├── feature_ablation.py           # 特征消融（学习曲线）
├── leave_one_oil_out.py          # 留一油类分析（增强版距离分析）
├── label_permutation_test.py     # 标签置换检验
├── generate_validation_report.py # 综合报告生成器
├── run_all_validations.py        # 运行所有验证
└── README.md                     # 本文件
```

## 使用方法

### 运行所有验证（推荐）

```bash
# 从项目根目录运行所有验证实验
python3 src/validation/run_all_validations.py
```

这将依次运行：
1. Leave-Replicate-Out Cross-Validation
2. Multi-Random-Seed Cross-Validation
3. Feature Ablation (Learning Curve)
4. Leave-One-Oil-Out (Distance-Based)
5. Label Permutation Test
6. 生成综合验证报告

### 运行单个验证

```bash
# 留一平行样本交叉验证
python3 src/validation/leave_replicate_out.py

# 多随机种子交叉验证
python3 src/validation/multi_seed_cv.py

# 特征消融（学习曲线）
python3 src/validation/feature_ablation.py

# 留一油类分析
python3 src/validation/leave_one_oil_out.py

# 标签置换检验
python3 src/validation/label_permutation_test.py

# 生成综合报告
python3 src/validation/generate_validation_report.py
```

## 验证方法

| 方法 | 结果 | 说明 |
|------|------|------|
| Leave-Replicate-Out | 100% | 泛化能力验证 - 证明模型学习的是油类反应模式而非记忆单个样本 |
| Multi-Seed CV | 95.4% ± 1.2% | 稳定性验证 - 窄置信区间表明性能稳定 |
| Feature Ablation | 50% → 96% | 传感器阵列价值 - 随酶数量增加准确率提升 |
| Leave-One-Oil-Out | 分离比率 4.32 | 类别可分性 - 左侧油类与训练类别边界分离良好 |
| Label Permutation | p < 0.001 | 统计显著性 - 证明模型学习的是真实的特征-标签关系 |

### 验证方法说明

#### 1. Leave-Replicate-Out CV
- **目的**：验证模型学习油类反应模式，而非记忆单个样本
- **策略**：每次留出一个完整的重复样本集（13个样本），用其余65个样本训练
- **结果解读**：>90% 表示强泛化能力

#### 2. Multi-Seed CV
- **目的**：验证模型在不同数据分割下的稳定性
- **策略**：20个随机种子，每个种子运行2折分层CV
- **结果解读**：95%置信区间宽度<5%表示高稳定性

#### 3. Feature Ablation
- **目的**：验证传感器阵列设计的价值
- **策略**：测试不同酶数量的子集（1, 3, 5, 10, 15个酶）
- **结果解读**：准确率随酶数量增加而提升，验证阵列设计合理性

#### 4. Leave-One-Oil-Out
- **目的**：验证模型识别新颖油类的能力（基于距离度量）
- **策略**：依次留出每个油类，计算左侧油类到训练类别的距离比
- **结果解读**：分离比>1.5表示左侧油类在训练类别边界外，分离良好

#### 5. Label Permutation Test
- **目的**：验证模型学习的是真实的特征-标签关系
- **策略**：打乱标签100次，构建零分布，与真实准确率比较
- **结果解读**：p < 0.001 表示极其显著，拒绝零假设

## 输出目录

所有结果保存在 `experiments/` 目录下：

```
experiments/
├── leave_replicate_out/
│   ├── metrics.json
│   └── confusion_matrix.png
├── multi_seed/
│   ├── metrics.json
│   └── stability_boxplot.png
├── feature_ablation/
│   ├── metrics.json
│   └── learning_curve.png
├── leave_one_oil_out/
│   ├── metrics.json
│   ├── enhanced_distance_analysis.png
│   └── leave_out_*.png (13个油类的可视化)
├── label_permutation/
│   ├── metrics.json
│   └── permutation_test.png
└── validation_report/
    ├── validation_summary.png (综合汇总图)
    └── validation_report.md (详细报告)
```

## 生成论文图表

运行所有验证后，综合报告包含：
- `validation_summary.png` - 6合1综合汇总图，适合论文使用
- `validation_report.md` - 详细的验证结果报告，包含所有统计数据和解读

```bash
# 仅生成报告（需要先运行过所有验证）
python3 src/validation/generate_validation_report.py
```

## 快速开始

```bash
# 1. 运行所有验证实验（约5-10分钟）
python3 src/validation/run_all_validations.py

# 2. 查看结果
ls experiments/validation_report/

# 3. 打开综合图
open experiments/validation_report/validation_summary.png
```
