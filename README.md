# YOLOv11 CoreML 完整项目

[![Platform](https://img.shields.io/badge/platform-iOS%20%7C%20macOS%20%7C%20Python-lightgrey)]()
[![iOS](https://img.shields.io/badge/iOS-15.0%2B-blue)]()
[![macOS](https://img.shields.io/badge/macOS-12.0%2B-blue)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

这是一个完整的 YOLOv11 CoreML 实现项目，提供从预训练 PyTorch 模型、CoreML 转换到跨平台 SDK 开发的完整解决方案。

## 🚀 项目特色

- **完整流程**: 从预训练 PyTorch 模型到 CoreML 部署的完整管道
- **🎯 精度保障**: 先进的IoU匹配算法和mAP计算，确保转换精度无损失
- **双平台支持**: Python 和 Swift SDK，支持服务器端和移动端部署
- **高性能**: 利用 Apple 神经引擎优化的 CoreML 推理
- **易于使用**: 简洁的 API 设计，丰富的文档和示例
- **生产就绪**: 包含完整的测试、基准测试和错误处理

## 📁 项目结构

```
yolo11_coreml/
├── 01_pytorch_setup/          # 第一步：PyTorch 环境搭建和模型测试
├── 02_coreml_conversion/       # 第二步：CoreML 模型转换和验证
├── 03_python_sdk/             # 第三步：Python SDK 封装
├── 04_swift_sdk/              # 第四步：Swift SDK 封装
├── shared_resources/          # 共享资源
│   ├── models/               # 模型文件
│   ├── test_images/          # 测试图片
│   └── docs/                 # 文档资源
└── README.md                 # 项目主文档
```

## 🔧 快速开始

### 方法一：完整流程（推荐）

按照项目逻辑顺序，从头开始体验完整的开发流程：

```bash
# 第一步：PyTorch 环境搭建和模型测试
cd 01_pytorch_setup
chmod +x setup_environment.sh
./setup_environment.sh
source venv/bin/activate
python test_pytorch_model.py

# 第二步：CoreML 模型转换和精度验证
cd ../02_coreml_conversion
pip install -r requirements.txt
python convert_and_validate.py
# 可选：运行详细精度分析
python accuracy_comparison.py --pytorch-model ../shared_resources/models/yolo11n.pt --coreml-model coreml_models/yolo11n_no_nms.mlpackage

# 第三步：Python SDK 测试
cd ../03_python_sdk
pip install -e .
python test_sdk.py

# 第四步：Swift SDK 集成（在 Xcode 中）
cd ../04_swift_sdk
# 在 Xcode 中打开并运行测试
```

### 方法二：直接使用 SDK

如果你已经有了 CoreML 模型，可以直接使用我们的 SDK：

#### Python SDK

```python
from yolo_sdk import YOLOv11CoreML

# 初始化模型
model = YOLOv11CoreML()

# 执行预测
results = model.predict('path/to/image.jpg')

# 或者获取解析后的结果
detections = model.predict_and_parse('path/to/image.jpg')
for detection in detections:
    print(f"{detection['class_name']}: {detection['confidence']:.2f}")
```

#### Swift SDK

```swift
import YOLOv11CoreMLSDK

// 初始化 SDK
let sdk = try YOLOv11CoreMLSDK()

// 执行检测
let detections = await sdk.detect(uiImage: yourImage)

// 处理结果
for detection in detections {
    print("\\(detection.label): \\(detection.confidence)")
}
```

## 📖 详细文档

### 分步指南

1. **[PyTorch 环境搭建](01_pytorch_setup/README.md)** - 环境配置和模型测试
2. **[CoreML 转换](02_coreml_conversion/README.md)** - 模型转换和验证流程
3. **[Python SDK](03_python_sdk/README.md)** - Python SDK 使用指南
4. **[Swift SDK](04_swift_sdk/README.md)** - Swift SDK 集成指南

### 技术文档

- **[完整项目指南](PROJECT_GUIDE.md)** - 详细的技术实现文档
- **[精度分析工具](02_coreml_conversion/accuracy_comparison.py)** - 专业的模型精度验证工具
- **[数据集管理器](02_coreml_conversion/dataset_manager.py)** - 测试数据集自动管理

## 🎯 使用场景

### 移动应用开发
- iOS/macOS 原生应用中的实时目标检测
- 相机应用中的物体识别
- AR 应用中的场景理解

### 服务器端部署
- Python Web 服务中的图像分析
- 批量图像处理任务
- API 服务中的目标检测

### 边缘计算
- 利用 Apple 神经引擎的高效推理
- 离线模式下的目标检测
- 低延迟的实时应用

## 📊 性能表现

| 设备 | 模型 | 平均推理时间 | FPS | 内存使用 |
|------|------|-------------|-----|---------|
| iPhone 14 Pro | YOLOv11n | ~15ms | ~65 | ~50MB |
| MacBook Pro M2 | YOLOv11n | ~8ms | ~125 | ~45MB |
| Python (CPU) | YOLOv11n | ~45ms | ~22 | ~200MB |

*实际性能可能因设备和使用场景而异*

## 🔄 支持的模型

- **YOLOv11n** - 纳米版本，速度最快
- **YOLOv11s** - 小型版本，平衡速度和精度
- **YOLOv11m** - 中型版本，更高精度
- **YOLOv11l** - 大型版本，最高精度
- **YOLOv11x** - 超大版本，最佳效果

## 🛠️ 环境要求

### Python 环境
- Python 3.8+
- PyTorch 2.0+
- Ultralytics 8.0+
- CoreMLTools 7.0+

### iOS/macOS 环境
- iOS 15.0+ / macOS 12.0+
- Xcode 14+
- Swift 5.5+

### 硬件建议
- 带有神经引擎的 Apple 芯片（A12+, M1+）获得最佳性能
- 8GB+ RAM 用于模型转换
- 充足的存储空间用于模型文件

## 🤝 贡献指南

我们欢迎社区贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 开发环境设置

```bash
# 克隆项目
git clone <repository-url>
cd yolo11_coreml

# 设置 Python 环境
cd 01_pytorch_setup
./setup_environment.sh
source venv/bin/activate

# 运行所有测试
python -m pytest tests/
```

## 📝 更新日志

### v1.0.0 (当前版本)
- ✅ 完整的 PyTorch 到 CoreML 转换流程
- ✅ **先进的精度对比系统** (IoU匹配 + mAP计算)
- ✅ Python SDK 与丰富的 API
- ✅ Swift SDK 支持 iOS/macOS
- ✅ 完整的测试和文档
- ✅ 性能基准测试工具
- ✅ 自动化数据集管理

### 计划功能
- 🔄 批量处理工具
- 🔄 模型量化优化
- 🔄 更多 YOLO 版本支持
- 🔄 Web 演示界面

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv11 实现
- [Apple CoreML](https://developer.apple.com/machine-learning/core-ml/) - 模型部署框架
- 所有为这个项目做出贡献的开发者

## 📞 支持

- 📧 Email: support@example.com
- 💬 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Wiki: [项目 Wiki](https://github.com/your-repo/wiki)

---

⭐ 如果这个项目对你有帮助，请给我们一个星标！