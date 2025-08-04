#!/usr/bin/env python3
"""
基础 YOLO 模型转换脚本
将 PyTorch (.pt) 模型转换为 CoreML (.mlpackage) 格式
"""

from ultralytics import YOLO
import os
import sys

def export_to_coreml(model_path, output_dir=None):
    """
    将 YOLO 模型导出为 CoreML 格式
    
    Args:
        model_path: 输入的 .pt 模型文件路径
        output_dir: 输出目录，默认为当前目录
    """
    try:
        print(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        print("开始导出 CoreML 模型...")
        model.export(format='coreml')
        
        # 获取生成的 mlpackage 文件名
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        mlpackage_path = f"{model_name}.mlpackage"
        
        # 如果指定了输出目录，移动文件
        if output_dir and output_dir != ".":
            os.makedirs(output_dir, exist_ok=True)
            import shutil
            new_path = os.path.join(output_dir, f"{model_name}.mlpackage")
            if os.path.exists(mlpackage_path):
                shutil.move(mlpackage_path, new_path)
                print(f"模型已导出到: {new_path}")
            else:
                print(f"警告: 未找到生成的模型文件 {mlpackage_path}")
        else:
            print(f"模型已导出: {mlpackage_path}")
            
    except Exception as e:
        print(f"导出失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python basic_export.py <model.pt> [output_dir]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        sys.exit(1)
    
    export_to_coreml(model_path, output_dir)