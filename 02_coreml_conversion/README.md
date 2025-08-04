# 第二步：CoreML 模型转换和验证

这是完整项目流程的第二步，主要目标是将 PyTorch 模型转换为 CoreML 格式，并进行完整的验证和精度对比。

## 🎯 目标

- 将 PyTorch 模型转换为 Apple CoreML 格式
- 验证转换后的模型功能正确性
- **🔍 精度对比分析** - 确保转换后精度无损失
- 性能基准测试和优化
- 生成详细的分析报告

## 📋 环境要求

- macOS 10.15+ (推荐 macOS 12+)
- Python 3.8+
- 已完成第一步的 PyTorch 环境搭建
- 8GB+ 可用内存用于模型转换

## 🚀 快速开始

### 一键转换和验证

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整转换流程（包含精度对比）
python convert_and_validate.py
```

### 单独运行精度分析

```bash
# 详细精度对比分析
python accuracy_comparison.py \
  --pytorch-model ../shared_resources/models/yolo11n.pt \
  --coreml-model coreml_models/yolo11n.mlpackage \
  --conf-thresholds 0.1 0.25 0.5
```

## 📁 文件说明

### 核心文件

- **`convert_and_validate.py`** - 一键转换和验证脚本
- **`accuracy_comparison.py`** - 🆕 精度对比分析工具
- **`requirements.txt`** - 环境依赖

### 转换工具集 (`scripts/`)

- **`basic_export.py`** - 基础转换脚本
- **`advanced_export.py`** - 高级批量转换工具
- **`validate_coreml.py`** - 模型验证脚本
- **`original_export.py`** - 原始简单脚本

### 输出目录

- **`coreml_models/`** - 转换后的 CoreML 模型
- **`accuracy_results/`** - 精度分析报告
- **`validation_results/`** - 验证结果图片

## 🎯 精度对比功能

### 为什么需要精度对比？

模型转换过程中可能出现精度损失的原因：
- 数值精度变化（float32 → float16）
- 操作符转换差异
- 量化和优化影响
- 不同推理引擎的实现差异

### 精度分析指标

我们的工具会计算以下关键指标：

1. **精确度 (Precision)**: 检测正确率
2. **召回率 (Recall)**: 目标发现率  
3. **F1 分数**: 精确度和召回率的调和平均
4. **IoU (Intersection over Union)**: 边界框重叠度
5. **置信度差异**: 两个模型输出置信度的差异

### 精度评估标准

- **✅ 优秀**: F1 分数 > 0.9，基本无精度损失
- **🟡 良好**: F1 分数 > 0.8，轻微精度损失
- **🟠 一般**: F1 分数 > 0.7，明显精度损失
- **❌ 较差**: F1 分数 ≤ 0.7，严重精度损失

## 🔍 使用示例

### 完整转换流程

```bash
python convert_and_validate.py
```

**输出示例**:
```
🚀 开始 CoreML 转换和验证流程
============================================================
✅ PyTorch 模型加载成功: ../shared_resources/models/yolo11n.pt
✅ 基础 CoreML 模型转换完成: coreml_models/yolo11n.mlpackage  
✅ 推理成功，检测到 4 个对象
✅ 优化模型保存: coreml_models/yolo11n_optimized.mlpackage

📊 性能对比结果:
   PyTorch 模型: 0.045s (22.2 FPS)
   CoreML 模型: 0.038s (26.3 FPS)  
   🚀 CoreML 模型快 15.6%

🎯 精度对比测试...
   PyTorch 检测数量: 4
   CoreML 检测数量: 4
   共同检测类别: 3
   平均置信度差异: 0.023
   ✅ 置信度一致性：优秀
   ✅ 检测数量：完全一致

🎉 CoreML 转换和验证完成！
```

### 详细精度分析

```bash
python accuracy_comparison.py \
  --pytorch-model ../shared_resources/models/yolo11n.pt \
  --coreml-model coreml_models/yolo11n.mlpackage
```

**生成的报告**:
- `accuracy_comparison_[timestamp].json` - 详细数值结果
- `accuracy_report_[timestamp].md` - 可读性分析报告

**报告示例**:
```markdown  
# YOLOv11 CoreML 精度对比报告

## 📊 总体结论
- **最佳置信度阈值**: 0.25
- **平均精确度**: 0.953 ± 0.028  
- **平均召回率**: 0.947 ± 0.035
- **平均 F1 分数**: 0.950 ± 0.025
- **平均 IoU**: 0.872 ± 0.045

✅ **精度评估**: 优秀 - CoreML 转换基本无精度损失
```

## 🛠️ 高级用法

### 批量转换多个模型

```bash
python scripts/advanced_export.py ../shared_resources/models/ -b -o coreml_models/
```

### 自定义转换参数

```python
from ultralytics import YOLO

model = YOLO('model.pt')
model.export(
    format='coreml',
    optimize=True,        # 启用优化
    nms=True,            # 包含 NMS
    simplify=True,       # 简化模型
    half=False           # 使用 float32 精度
)
```

### 指定置信度阈值测试

```bash
python accuracy_comparison.py \
  --pytorch-model model.pt \
  --coreml-model model.mlpackage \
  --conf-thresholds 0.1 0.2 0.3 0.4 0.5
```

## 🐛 常见问题

### 1. 精度损失严重

**可能原因**:
- 模型量化过度
- 不支持的操作符转换错误

**解决方案**:
```python
# 禁用量化，使用 float32
model.export(format='coreml', optimize=False, half=False)
```

### 2. 转换失败

**检查项**:
- PyTorch 和 CoreMLTools 版本兼容性
- 模型是否包含不支持的操作
- 系统内存是否充足

### 3. 性能不如预期

**优化建议**:
```python
# 启用所有优化选项
model.export(
    format='coreml',
    optimize=True,
    simplify=True,
    nms=True
)
```

### 4. 精度分析报告为空

**检查项**:
- 测试图片是否包含可检测对象
- 置信度阈值是否过高
- 网络连接是否正常（下载测试图片）

## 📊 性能基准

| 设备 | PyTorch (FPS) | CoreML (FPS) | 精度保持 |
|------|---------------|--------------|----------|
| MacBook Pro M2 | ~22 | ~26 | >95% |
| MacBook Air M1 | ~18 | ~23 | >95% |
| Intel Mac | ~12 | ~15 | >95% |

## ➡️ 下一步

完成这一步后，你应该得到：

- ✅ 工作的 CoreML 模型文件
- ✅ 验证的推理功能
- ✅ **精度对比报告** 
- ✅ 性能基准数据

继续进入下一步：

```bash
cd ../03_python_sdk
```

阅读 [Python SDK 开发指南](../03_python_sdk/README.md)。