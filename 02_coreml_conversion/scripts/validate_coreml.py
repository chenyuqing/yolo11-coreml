#!/usr/bin/env python3
"""
CoreML 模型验证脚本
验证转换后的 CoreML 模型是否正常工作
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import coremltools as ct
from PIL import Image
import requests
from io import BytesIO

def download_test_image(url="https://ultralytics.com/images/bus.jpg"):
    """下载测试图片"""
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"下载测试图片失败: {e}")
        return None

def validate_coreml_model(model_path, test_image_path=None):
    """验证 CoreML 模型"""
    print(f"验证模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return False
    
    try:
        # 1. 使用 coremltools 加载模型
        print("1. 使用 coremltools 验证...")
        ct_model = ct.models.MLModel(model_path)
        spec = ct_model.get_spec()
        
        print(f"   - 模型版本: {spec.specificationVersion}")
        print(f"   - 输入: {[input.name for input in spec.description.input]}")
        print(f"   - 输出: {[output.name for output in spec.description.output]}")
        
        # 2. 使用 YOLO 加载模型
        print("2. 使用 YOLO 验证...")
        model = YOLO(model_path)
        
        # 3. 测试推理
        print("3. 测试推理...")
        
        # 准备测试图片
        if test_image_path and os.path.exists(test_image_path):
            test_source = test_image_path
            print(f"   使用测试图片: {test_image_path}")
        else:
            # 尝试下载默认测试图片
            print("   下载默认测试图片...")
            test_image = download_test_image()
            if test_image:
                test_image.save("temp_test_image.jpg")
                test_source = "temp_test_image.jpg"
            else:
                test_source = "https://ultralytics.com/images/bus.jpg"
        
        # 执行推理
        results = model.predict(test_source, verbose=False)
        
        if results:
            result = results[0]
            detections = len(result.boxes) if result.boxes is not None else 0
            print(f"   ✓ 推理成功，检测到 {detections} 个对象")
            
            # 清理临时文件
            if os.path.exists("temp_test_image.jpg"):
                os.remove("temp_test_image.jpg")
            
            return True
        else:
            print("   ✗ 推理失败")
            return False
            
    except Exception as e:
        print(f"验证失败: {e}")
        return False

def batch_validate(models_dir, pattern="*.mlpackage"):
    """批量验证模型"""
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"错误: 目录不存在 {models_dir}")
        return
    
    model_files = list(models_path.glob(pattern))
    if not model_files:
        print(f"未找到匹配的模型文件: {pattern}")
        return
    
    print(f"找到 {len(model_files)} 个模型文件")
    print("-" * 50)
    
    success_count = 0
    for model_file in model_files:
        print(f"\n处理: {model_file.name}")
        if validate_coreml_model(str(model_file)):
            success_count += 1
        print("-" * 50)
    
    print(f"\n批量验证完成: {success_count}/{len(model_files)} 通过")

def main():
    parser = argparse.ArgumentParser(description="CoreML 模型验证工具")
    parser.add_argument("input", help="模型文件或目录")
    parser.add_argument("-t", "--test-image", help="测试图片路径")
    parser.add_argument("-b", "--batch", action="store_true", help="批量验证模式")
    parser.add_argument("--pattern", default="*.mlpackage", help="批量模式下的文件模式")
    
    args = parser.parse_args()
    
    if args.batch:
        batch_validate(args.input, args.pattern)
    else:
        success = validate_coreml_model(args.input, args.test_image)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()