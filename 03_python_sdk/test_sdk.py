#!/usr/bin/env python3
"""
第三步：Python SDK 测试脚本
测试 YOLOv11 CoreML Python SDK 的功能
"""

import sys
import os
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    
    try:
        import numpy as np
        print(f"   ✅ NumPy: {np.__version__}")
    except ImportError:
        print("   ❌ NumPy 未安装")
        return False
    
    try:
        from ultralytics import YOLO, __version__
        print(f"   ✅ Ultralytics: {__version__}")
    except ImportError:
        print("   ❌ Ultralytics 未安装")
        return False
    
    try:
        from yolo_sdk import YOLOv11CoreML, __version__
        print(f"   ✅ YOLO SDK: {__version__}")
    except ImportError as e:
        print(f"   ❌ YOLO SDK 导入失败: {e}")
        return False
    
    return True

def test_model_loading():
    """测试模型加载"""
    print("\n🔄 测试模型加载...")
    
    try:
        from yolo_sdk import YOLOv11CoreML
        
        # 测试自动查找模型
        print("   测试自动查找模型...")
        model = YOLOv11CoreML()
        print(f"   ✅ 模型加载成功: {model.model_path}")
        
        return model
        
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return None

def test_model_info(model):
    """测试模型信息获取"""
    print("\n📋 测试模型信息...")
    
    try:
        info = model.info()
        print("   模型信息:")
        for key, value in info.items():
            if key == 'class_names':
                print(f"     {key}: {len(value)} 个类别")
            else:
                print(f"     {key}: {value}")
        
        # 显示部分类别名称
        class_names = model.class_names
        if len(class_names) > 0:
            print("   前10个类别:")
            for i, (class_id, class_name) in enumerate(list(class_names.items())[:10]):
                print(f"     {class_id}: {class_name}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 获取模型信息失败: {e}")
        return False

def test_basic_prediction(model):
    """测试基础预测功能"""
    print("\n🔍 测试基础预测...")
    
    # 准备测试图片
    test_image_path = project_root / "shared_resources" / "test_images" / "bus.jpg"
    
    if not test_image_path.exists():
        print("   使用在线测试图片...")
        test_source = "https://ultralytics.com/images/bus.jpg"
    else:
        print(f"   使用本地测试图片: {test_image_path}")
        test_source = str(test_image_path)
    
    try:
        # 执行预测
        results = model.predict(test_source, save=True, save_dir="03_python_sdk/results")
        
        if results:
            result = results[0]
            if result.boxes is not None:
                num_detections = len(result.boxes)
                print(f"   ✅ 预测成功！检测到 {num_detections} 个对象")
                
                # 显示检测详情
                for i, box in enumerate(result.boxes):
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = model.class_names[cls]
                    print(f"     对象 {i+1}: {class_name} (置信度: {conf:.2f})")
            else:
                print("   ✅ 预测成功，但未检测到对象")
            
            return True
        else:
            print("   ❌ 预测失败，无结果返回")
            return False
            
    except Exception as e:
        print(f"   ❌ 预测失败: {e}")
        return False

def test_parsed_prediction(model):
    """测试解析后的预测结果"""
    print("\n📊 测试解析预测结果...")
    
    test_image_path = project_root / "shared_resources" / "test_images" / "bus.jpg"
    test_source = str(test_image_path) if test_image_path.exists() else "https://ultralytics.com/images/bus.jpg"
    
    try:
        parsed_results = model.predict_and_parse(test_source, conf=0.3)
        
        print(f"   ✅ 解析完成！共 {len(parsed_results)} 个检测结果")
        
        for i, detection in enumerate(parsed_results):
            print(f"   检测结果 {i+1}:")
            print(f"     类别: {detection['class_name']} (ID: {detection['class_id']})")
            print(f"     置信度: {detection['confidence']:.3f}")
            bbox = detection['bbox']
            print(f"     边界框: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 解析预测失败: {e}")
        return False

def test_different_confidence_thresholds(model):
    """测试不同置信度阈值"""
    print("\n🔧 测试不同置信度阈值...")
    
    test_image_path = project_root / "shared_resources" / "test_images" / "bus.jpg"
    test_source = str(test_image_path) if test_image_path.exists() else "https://ultralytics.com/images/bus.jpg"
    
    confidence_levels = [0.1, 0.25, 0.5, 0.7, 0.9]
    
    for conf in confidence_levels:
        try:
            parsed_results = model.predict_and_parse(test_source, conf=conf)
            print(f"   置信度 {conf}: {len(parsed_results)} 个检测结果")
        except Exception as e:
            print(f"   置信度 {conf}: 失败 - {e}")

def test_benchmark(model):
    """测试性能基准"""
    print("\n⏱️  测试性能基准...")
    
    test_image_path = project_root / "shared_resources" / "test_images" / "bus.jpg"
    test_source = str(test_image_path) if test_image_path.exists() else "https://ultralytics.com/images/bus.jpg"
    
    try:
        benchmark_results = model.benchmark(test_source, num_runs=5, warmup_runs=2)
        
        print("   基准测试结果:")
        print(f"     平均时间: {benchmark_results['avg_time']:.3f}s")
        print(f"     标准差: {benchmark_results['std_time']:.3f}s")
        print(f"     最快: {benchmark_results['min_time']:.3f}s")
        print(f"     最慢: {benchmark_results['max_time']:.3f}s")
        print(f"     平均 FPS: {benchmark_results['avg_fps']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 基准测试失败: {e}")
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n🛡️  测试错误处理...")
    
    try:
        from yolo_sdk import YOLOv11CoreML
        
        # 测试不存在的模型路径
        print("   测试不存在的模型路径...")
        try:
            model = YOLOv11CoreML(model_path="nonexistent_model.mlpackage")
            print("   ❌ 应该抛出异常但没有")
            return False
        except FileNotFoundError:
            print("   ✅ 正确处理了不存在的模型路径")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 错误处理测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 YOLOv11 CoreML Python SDK 测试")
    print("=" * 50)
    
    # 1. 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败")
        return False
    
    # 2. 测试模型加载
    model = test_model_loading()
    if model is None:
        print("❌ 模型加载失败，退出测试")
        return False
    
    # 3. 测试模型信息
    if not test_model_info(model):
        print("⚠️  模型信息获取失败")
    
    # 4. 测试基础预测
    if not test_basic_prediction(model):
        print("❌ 基础预测测试失败")
        return False
    
    # 5. 测试解析预测
    if not test_parsed_prediction(model):
        print("⚠️  解析预测测试失败")
    
    # 6. 测试不同置信度阈值
    test_different_confidence_thresholds(model)
    
    # 7. 测试性能基准
    if not test_benchmark(model):
        print("⚠️  性能基准测试失败")
    
    # 8. 测试错误处理
    if not test_error_handling():
        print("⚠️  错误处理测试失败")
    
    print("\n" + "=" * 50)
    print("🎉 Python SDK 测试完成！")
    print("\n📁 测试结果保存在: 03_python_sdk/results/")
    print("📋 下一步: 运行 04_swift_sdk 创建 Swift SDK")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)