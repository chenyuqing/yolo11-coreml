#!/usr/bin/env python3
"""
第一步：PyTorch YOLO11 模型测试脚本
验证本地环境和模型是否正常工作
"""

import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查 Python 版本
    python_version = sys.version_info
    print(f"   Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查 PyTorch
    print(f"   PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 设备: {torch.cuda.get_device_name(0)}")
    
    # 检查 Ultralytics
    try:
        from ultralytics import __version__
        print(f"   Ultralytics 版本: {__version__}")
    except:
        print("   ⚠️  Ultralytics 版本信息获取失败")
    
    print("✅ 环境检查完成\n")

def download_model():
    """下载或验证模型文件"""
    model_path = project_root / "shared_resources" / "models" / "yolo11n.pt"
    
    if not model_path.exists():
        print("📥 模型文件不存在，正在下载...")
        model = YOLO('yolo11n.pt')  # 自动下载
        
        # 移动下载的模型到指定位置
        downloaded_path = Path("yolo11n.pt")
        if downloaded_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path.rename(model_path)
            print(f"   模型已保存到: {model_path}")
    else:
        print(f"✅ 模型文件存在: {model_path}")
    
    return model_path

def prepare_test_image():
    """准备测试图片"""
    # 首先检查本地测试图片
    local_image = project_root / "shared_resources" / "test_images" / "bus.jpg"
    
    if local_image.exists():
        print(f"✅ 使用本地测试图片: {local_image}")
        return str(local_image)
    
    # 下载测试图片
    print("📥 下载测试图片...")
    try:
        url = "https://ultralytics.com/images/bus.jpg"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # 保存图片
        local_image.parent.mkdir(parents=True, exist_ok=True)
        with open(local_image, 'wb') as f:
            f.write(response.content)
        
        print(f"   测试图片已保存: {local_image}")
        return str(local_image)
        
    except Exception as e:
        print(f"   ⚠️  下载失败: {e}")
        # 返回在线URL作为备用
        return "https://ultralytics.com/images/bus.jpg"

def test_model_loading(model_path):
    """测试模型加载"""
    print("🔄 测试模型加载...")
    
    try:
        model = YOLO(model_path)
        print("✅ 模型加载成功")
        
        # 显示模型信息
        print("\n📋 模型信息:")
        model.info()
        
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def test_inference(model, image_source):
    """测试推理功能"""
    print("\n🔍 测试推理功能...")
    
    try:
        # 执行推理
        results = model.predict(
            source=image_source, 
            save=True,
            save_dir="01_pytorch_setup/results",
            verbose=False
        )
        
        if results:
            result = results[0]
            
            # 分析结果
            if result.boxes is not None:
                num_detections = len(result.boxes)
                print(f"✅ 推理成功！检测到 {num_detections} 个对象")
                
                # 显示检测详情
                for i, box in enumerate(result.boxes):
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = model.names[cls]
                    print(f"   对象 {i+1}: {class_name} (置信度: {conf:.2f})")
            else:
                print("✅ 推理成功，但未检测到对象")
            
            return True
        else:
            print("❌ 推理失败，无结果返回")
            return False
            
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return False

def test_different_sources(model):
    """测试不同输入源"""
    print("\n🔍 测试不同输入源...")
    
    test_sources = [
        ("本地图片", project_root / "shared_resources" / "test_images" / "bus.jpg"),
        ("在线图片", "https://ultralytics.com/images/bus.jpg"),
    ]
    
    for source_name, source_path in test_sources:
        if source_name == "本地图片" and not Path(source_path).exists():
            continue
            
        print(f"   测试 {source_name}...")
        try:
            results = model.predict(source_path, verbose=False)
            if results and results[0].boxes is not None:
                num_detections = len(results[0].boxes)
                print(f"   ✅ {source_name}: 检测到 {num_detections} 个对象")
            else:
                print(f"   ✅ {source_name}: 推理成功，无对象检测")
        except Exception as e:
            print(f"   ❌ {source_name}: 失败 - {e}")

def benchmark_performance(model, image_source, num_runs=10):
    """性能基准测试"""
    print(f"\n⏱️  性能基准测试 ({num_runs} 次运行)...")
    
    import time
    times = []
    
    # 预热
    model.predict(image_source, verbose=False)
    
    # 基准测试
    for i in range(num_runs):
        start_time = time.time()
        model.predict(image_source, verbose=False)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time
    
    print(f"   平均推理时间: {avg_time:.3f}s (±{std_time:.3f}s)")
    print(f"   平均 FPS: {fps:.1f}")
    print(f"   最快: {min(times):.3f}s, 最慢: {max(times):.3f}s")

def main():
    """主函数"""
    print("🚀 YOLOv11 PyTorch 模型测试")
    print("=" * 50)
    
    # 1. 检查环境
    check_environment()
    
    # 2. 准备模型和测试数据
    model_path = download_model()
    image_source = prepare_test_image()
    
    # 3. 测试模型加载
    model = test_model_loading(model_path)
    if model is None:
        print("❌ 模型加载失败，退出测试")
        return False
    
    # 4. 测试基础推理
    if not test_inference(model, image_source):
        print("❌ 基础推理测试失败")
        return False
    
    # 5. 测试不同输入源
    test_different_sources(model)
    
    # 6. 性能基准测试
    benchmark_performance(model, image_source)
    
    print("\n" + "=" * 50)
    print("🎉 PyTorch 模型测试完成！")
    print("\n📁 结果文件保存在: 01_pytorch_setup/results/")
    print("📋 下一步: 运行 02_coreml_conversion 进行模型转换")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)