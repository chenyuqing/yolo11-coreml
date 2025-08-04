#!/usr/bin/env python3
"""
高级 YOLO 模型转换脚本
支持批量转换、自定义参数和验证
"""

from ultralytics import YOLO
import os
import sys
import argparse
import shutil
from pathlib import Path
import time

class YOLOCoreMLConverter:
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def log(self, message):
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def validate_model(self, model_path):
        """验证模型文件是否有效"""
        try:
            model = YOLO(model_path)
            self.log(f"✓ 模型验证通过: {model_path}")
            return True
        except Exception as e:
            self.log(f"✗ 模型验证失败: {e}")
            return False
    
    def export_single_model(self, model_path, output_dir, optimize=True):
        """导出单个模型"""
        try:
            self.log(f"开始处理: {model_path}")
            
            # 验证模型
            if not self.validate_model(model_path):
                return False
            
            model = YOLO(model_path)
            
            # 导出参数
            export_kwargs = {
                'format': 'coreml',
                'optimize': optimize,
                'nms': True,  # 启用 NMS
            }
            
            self.log("正在导出 CoreML 模型...")
            model.export(**export_kwargs)
            
            # 处理输出文件
            model_name = Path(model_path).stem
            mlpackage_name = f"{model_name}.mlpackage"
            
            # 移动到指定目录
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                dest_path = Path(output_dir) / mlpackage_name
                
                if Path(mlpackage_name).exists():
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.move(mlpackage_name, dest_path)
                    self.log(f"✓ 导出完成: {dest_path}")
                else:
                    self.log(f"✗ 未找到生成的模型文件: {mlpackage_name}")
                    return False
            else:
                self.log(f"✓ 导出完成: {mlpackage_name}")
            
            return True
            
        except Exception as e:
            self.log(f"✗ 导出失败: {e}")
            return False
    
    def batch_export(self, input_dir, output_dir, pattern="*.pt"):
        """批量导出模型"""
        input_path = Path(input_dir)
        if not input_path.exists():
            self.log(f"✗ 输入目录不存在: {input_dir}")
            return
        
        model_files = list(input_path.glob(pattern))
        if not model_files:
            self.log(f"✗ 未找到匹配的模型文件: {pattern}")
            return
        
        self.log(f"找到 {len(model_files)} 个模型文件")
        
        success_count = 0
        for model_file in model_files:
            if self.export_single_model(str(model_file), output_dir):
                success_count += 1
        
        self.log(f"批量转换完成: {success_count}/{len(model_files)} 成功")

def main():
    parser = argparse.ArgumentParser(description="YOLO CoreML 高级转换工具")
    parser.add_argument("input", help="输入模型文件或目录")
    parser.add_argument("-o", "--output", help="输出目录")
    parser.add_argument("-b", "--batch", action="store_true", help="批量处理模式")
    parser.add_argument("--pattern", default="*.pt", help="批量模式下的文件模式")
    parser.add_argument("--no-optimize", action="store_true", help="禁用优化")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    
    args = parser.parse_args()
    
    converter = YOLOCoreMLConverter(verbose=not args.quiet)
    
    if args.batch:
        converter.batch_export(args.input, args.output, args.pattern)
    else:
        if not os.path.exists(args.input):
            print(f"错误: 文件不存在 {args.input}")
            sys.exit(1)
        
        success = converter.export_single_model(
            args.input, 
            args.output, 
            optimize=not args.no_optimize
        )
        
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()