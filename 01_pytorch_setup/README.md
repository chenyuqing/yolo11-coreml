# 第一步：PyTorch 环境搭建和模型测试

这是完整项目流程的第一步，主要目标是建立 Python 环境并验证 YOLOv11 PyTorch 模型的基本功能。

## 🎯 目标

- 搭建完整的 PyTorch 开发环境
- 下载和验证 YOLOv11 预训练模型
- 测试模型的推理功能
- 进行性能基准测试
- 为后续 CoreML 转换做准备

## 📋 环境要求

- Python 3.8+
- 至少 4GB 可用内存
- 稳定的网络连接（用于下载模型）

## 🚀 快速开始

### 自动化环境搭建

```bash
# 给脚本添加执行权限
chmod +x setup_environment.sh

# 运行环境搭建脚本
./setup_environment.sh

# 激活虚拟环境
source venv/bin/activate

# 运行模型测试
python test_pytorch_model.py
```

### 手动环境搭建

如果自动化脚本遇到问题，可以手动执行以下步骤：

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 升级 pip
pip install --upgrade pip

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行测试
python test_pytorch_model.py
```

## 📁 文件说明

### 核心文件

- **`setup_environment.sh`** - 自动化环境搭建脚本
- **`test_pytorch_model.py`** - 完整的模型测试脚本
- **`requirements.txt`** - Python 依赖列表

### 测试脚本功能

`test_pytorch_model.py` 包含以下测试功能：

1. **环境检查**
   - Python 版本验证
   - PyTorch 和 CUDA 状态
   - Ultralytics 版本检查

2. **模型管理**
   - 自动下载 YOLOv11n 模型
   - 模型文件验证和加载
   - 模型信息展示

3. **推理测试**
   - 本地图片推理
   - 在线图片推理
   - 不同输入源测试

4. **性能基准**
   - 多次推理时间统计
   - FPS 计算
   - 性能稳定性分析

## 🔍 测试输出示例

成功运行后，你会看到类似以下的输出：

```
🚀 YOLOv11 PyTorch 模型测试
==================================================
🔍 检查环境配置...
   Python 版本: 3.9.7
   PyTorch 版本: 2.1.0
   CUDA 可用: True
   CUDA 版本: 11.8
   GPU 设备: NVIDIA GeForce RTX 3080
   Ultralytics 版本: 8.0.20
✅ 环境检查完成

✅ 模型文件存在: /path/to/shared_resources/models/yolo11n.pt

🔄 测试模型加载...
✅ 模型加载成功

📋 模型信息:
Model summary: 225 layers, 3157200 parameters, 0 gradients, 8.9 GFLOPs

🔍 测试推理功能...
✅ 推理成功！检测到 4 个对象
   对象 1: bus (置信度: 0.89)
   对象 2: person (置信度: 0.76)
   对象 3: person (置信度: 0.63)
   对象 4: handbag (置信度: 0.52)

⏱️  性能基准测试 (10 次运行)...
   平均推理时间: 0.045s (±0.003s)
   平均 FPS: 22.3
   最快: 0.041s, 最慢: 0.051s

==================================================
🎉 PyTorch 模型测试完成！

📁 结果文件保存在: 01_pytorch_setup/results/
📋 下一步: 运行 02_coreml_conversion 进行模型转换
```

## 📊 预期结果

成功完成这一步后，你应该得到：

1. **工作的 Python 环境**
   - 正确安装的虚拟环境
   - 所有必需的依赖包

2. **验证的模型文件**
   - 下载的 `yolo11n.pt` 模型
   - 保存在 `shared_resources/models/` 目录

3. **测试结果**
   - 推理结果图片（保存在 `results/` 目录）
   - 性能基准数据
   - 模型信息报告

## 🐛 常见问题

### 1. 网络连接问题

**问题**: 模型下载失败
```
ConnectionError: Failed to download model
```

**解决方案**:
```bash
# 手动下载模型
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
mv yolo11n.pt ../shared_resources/models/
```

### 2. 依赖安装问题

**问题**: PyTorch 安装失败

**解决方案**:
```bash
# 根据你的系统选择合适的 PyTorch 版本
# CPU 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 版本（根据你的 CUDA 版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 内存不足

**问题**: 内存不足导致测试失败

**解决方案**:
- 关闭其他占用内存的程序
- 减少基准测试的运行次数
- 使用更小的输入图片

### 4. 权限问题

**问题**: 脚本没有执行权限

**解决方案**:
```bash
chmod +x setup_environment.sh
```

## 📈 性能优化建议

### GPU 加速

如果你有 NVIDIA GPU：

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 验证 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"
```

### 环境变量优化

```bash
# 设置 PyTorch 线程数
export OMP_NUM_THREADS=4

# 优化 CUDA 设置
export CUDA_LAUNCH_BLOCKING=1
```

## ➡️ 下一步

完成这一步后，你可以进入下一步：

```bash
cd ../02_coreml_conversion
```

确保你已经：
- ✅ 成功运行了 PyTorch 模型测试
- ✅ 模型文件保存在正确位置
- ✅ 获得了满意的性能结果

继续阅读 [CoreML 转换指南](../02_coreml_conversion/README.md)。