# 第三步：Python SDK 开发

这是完整项目流程的第三步，主要目标是将 CoreML 模型封装成易于使用的 Python SDK，提供简洁的 API 接口。

## 🎯 目标

- 封装 CoreML 模型为 Python SDK
- 提供直观的 API 接口
- 支持多种输入格式（本地文件、URL、PIL图片）
- 包含完整的结果解析和后处理
- 提供性能基准测试工具

## 📋 环境要求

- Python 3.8+
- 已完成前两步的模型转换
- macOS 系统（CoreML 运行要求）

## 🚀 快速开始

### 安装 SDK

```bash
# 进入 SDK 目录
cd 03_python_sdk

# 安装 SDK（开发模式）
pip install -e .

# 运行测试
python test_sdk.py
```

### 基本使用

```python
from yolo_sdk import YOLOv11CoreML

# 初始化模型（自动查找模型文件）
model = YOLOv11CoreML()

# 方法1：获取原始推理结果
results = model.predict('path/to/image.jpg')
print(results)

# 方法2：获取解析后的检测结果
detections = model.predict_and_parse('path/to/image.jpg')
for detection in detections:
    print(f"{detection['class_name']}: {detection['confidence']:.2f}")

# 方法3：包含基准测试的预测
results, benchmark = model.predict_with_benchmark('path/to/image.jpg')
print(f"推理时间: {benchmark['inference_time']:.3f}s")
```

## 📁 文件结构

```
03_python_sdk/
├── src/
│   └── yolo_sdk/
│       └── __init__.py          # SDK 主文件
├── models/                      # 模型文件目录
├── test_sdk.py                  # SDK 测试脚本
├── setup.cfg                    # 包配置
├── pyproject.toml              # 项目配置
├── MANIFEST.in                 # 包含文件配置
└── README.md                   # 使用文档
```

## 🔧 SDK API 文档

### YOLOv11CoreML 类

#### 初始化

```python
model = YOLOv11CoreML(
    model_path=None,        # 模型路径，None 时自动查找
    confidence=0.25,        # 置信度阈值
    iou_threshold=0.45      # NMS IoU 阈值
)
```

#### 主要方法

##### `predict(source, **kwargs)`
执行目标检测，返回原始结果。

**参数:**
- `source`: 图片源（文件路径、URL或PIL图片）
- `**kwargs`: 传递给 YOLO 的其他参数

**返回:** Ultralytics Results 对象

##### `predict_and_parse(source, **kwargs)`
执行检测并解析结果为易用格式。

**返回格式:**
```python
[
    {
        'class_id': 0,
        'class_name': 'person',
        'confidence': 0.85,
        'bbox': [x1, y1, x2, y2],
        'area': 1250.5
    },
    # ... 更多检测结果
]
```

##### `predict_with_benchmark(source, **kwargs)`
执行检测并包含性能基准。

**返回:**
```python
results, {
    'inference_time': 0.045,    # 推理时间（秒）
    'fps': 22.2,               # 帧率
    'model_load_time': 0.123   # 模型加载时间
}
```

##### `benchmark(source, runs=10)`
运行性能基准测试。

**返回:**
```python
{
    'avg_time': 0.045,         # 平均时间
    'std_time': 0.003,         # 时间标准差
    'min_time': 0.041,         # 最短时间
    'max_time': 0.051,         # 最长时间
    'avg_fps': 22.3,           # 平均FPS
    'total_runs': 10           # 总运行次数
}
```

## 📊 使用示例

### 基础目标检测

```python
from yolo_sdk import YOLOv11CoreML
from PIL import Image

# 初始化模型
detector = YOLOv11CoreML()

# 检测本地图片
detections = detector.predict_and_parse('test_image.jpg')

# 打印结果
print(f"检测到 {len(detections)} 个对象:")
for det in detections:
    print(f"  {det['class_name']}: {det['confidence']:.2f}")
```

### 在线图片检测

```python
# 检测在线图片
url = "https://ultralytics.com/images/bus.jpg"
detections = detector.predict_and_parse(url)

for det in detections:
    bbox = det['bbox']
    print(f"{det['class_name']}: {det['confidence']:.2f} at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
```

### 性能基准测试

```python
# 运行基准测试
benchmark = detector.benchmark('test_image.jpg', runs=20)

print("性能基准测试结果:")
print(f"  平均推理时间: {benchmark['avg_time']:.3f}s (±{benchmark['std_time']:.3f}s)")
print(f"  平均 FPS: {benchmark['avg_fps']:.1f}")
print(f"  最快: {benchmark['min_time']:.3f}s, 最慢: {benchmark['max_time']:.3f}s")
```

### 批量处理

```python
import glob

# 处理文件夹中的所有图片
image_files = glob.glob("images/*.jpg")

for image_file in image_files:
    detections = detector.predict_and_parse(image_file)
    print(f"{image_file}: {len(detections)} 个对象")
```

## 🐛 常见问题

### 1. 模型文件未找到

**错误:** `FileNotFoundError: No CoreML model found`

**解决方案:**
```python
# 指定模型路径
model = YOLOv11CoreML(model_path='path/to/your/model.mlpackage')
```

### 2. CoreML 不支持

**错误:** `CoreML is not available on this platform`

**解决方案:** SDK 仅支持 macOS 系统

### 3. 推理速度慢

**解决方案:**
- 确保使用带有神经引擎的 Apple 芯片
- 检查模型是否正确优化
- 减少输入图片尺寸

## 📈 性能优化

### 模型预热

```python
# 预热模型以获得更稳定的性能
detector = YOLOv11CoreML()

# 预热（使用小图片）
import numpy as np
from PIL import Image
warm_up_image = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
detector.predict(warm_up_image)

# 现在进行实际检测
results = detector.predict('real_image.jpg')
```

### 批量处理优化

```python
# 复用模型实例
detector = YOLOv11CoreML()

for image_file in image_files:
    # 避免重复初始化
    results = detector.predict(image_file)
```

## 🔧 高级配置

### 自定义置信度阈值

```python
# 初始化时设置
detector = YOLOv11CoreML(confidence=0.5)

# 或在预测时设置
results = detector.predict('image.jpg', conf=0.3)
```

### 指定特定类别

```python
# 只检测人和车
results = detector.predict('image.jpg', classes=[0, 2])  # person, car
```

## ➡️ 下一步

完成这一步后，你应该得到：

- ✅ 可用的 Python SDK
- ✅ 验证的 API 功能
- ✅ 性能基准数据

继续进入下一步：

```bash
cd ../04_swift_sdk
```

阅读 [Swift SDK 开发指南](../04_swift_sdk/README.md)。