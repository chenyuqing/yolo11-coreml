#!/usr/bin/env python3
"""
第二步：CoreML 模型转换和验证完整流程
"""

import os
import sys
import time
from pathlib import Path
import shutil
from ultralytics import YOLO
import coremltools as ct
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class CoreMLConverter:
    def __init__(self):
        self.project_root = project_root
        self.models_dir = self.project_root / "shared_resources" / "models"
        self.output_dir = Path("02_coreml_conversion") / "coreml_models"
        self.output_dir.mkdir(exist_ok=True)
        
    def log(self, message, level="INFO"):
        """日志输出"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def check_pytorch_model(self):
        """检查 PyTorch 模型"""
        self.log("检查 PyTorch 模型...")
        
        pt_model_path = self.models_dir / "yolo11n.pt"
        if not pt_model_path.exists():
            self.log(f"❌ PyTorch 模型不存在: {pt_model_path}", "ERROR")
            self.log("请先运行 01_pytorch_setup/test_pytorch_model.py", "ERROR")
            return None
        
        try:
            model = YOLO(pt_model_path)
            self.log(f"✅ PyTorch 模型加载成功: {pt_model_path}")
            return model, pt_model_path
        except Exception as e:
            self.log(f"❌ PyTorch 模型加载失败: {e}", "ERROR")
            return None
    
    def convert_to_coreml(self, model, pt_model_path):
        """转换为 CoreML"""
        self.log("开始转换为 CoreML...")
        
        try:
            # 基础转换
            self.log("执行基础转换...")
            model.export(
                format='coreml',
                optimize=True,
                nms=True
            )
            
            # 处理生成的文件
            model_name = pt_model_path.stem
            mlpackage_name = f"{model_name}.mlpackage"
            
            if Path(mlpackage_name).exists():
                # 移动到输出目录
                dest_path = self.output_dir / mlpackage_name
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.move(mlpackage_name, dest_path)
                
                self.log(f"✅ 基础 CoreML 模型转换完成: {dest_path}")
                return dest_path
            else:
                self.log(f"❌ 转换失败，未找到生成的模型: {mlpackage_name}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"❌ CoreML 转换失败: {e}", "ERROR")
            return None
    
    def optimize_coreml_model(self, coreml_path):
        """优化 CoreML 模型"""
        self.log("优化 CoreML 模型...")
        
        try:
            # 加载模型
            model = ct.models.MLModel(str(coreml_path))
            
            # 应用优化
            self.log("应用权重量化...")
            model = ct.models.neural_network.quantization_utils.quantize_weights(
                model, 
                nbits=16
            )
            
            # 保存优化后的模型
            optimized_path = coreml_path.parent / f"{coreml_path.stem}_optimized.mlpackage"
            model.save(str(optimized_path))
            
            self.log(f"✅ 优化模型保存: {optimized_path}")
            return optimized_path
            
        except Exception as e:
            self.log(f"⚠️  模型优化失败: {e}", "WARNING")
            return None
    
    def validate_coreml_model(self, coreml_path):
        """验证 CoreML 模型"""
        self.log(f"验证 CoreML 模型: {coreml_path.name}")
        
        try:
            # 1. 使用 coremltools 验证
            self.log("1. 使用 coremltools 验证...")
            ct_model = ct.models.MLModel(str(coreml_path))
            spec = ct_model.get_spec()
            
            self.log(f"   模型版本: {spec.specificationVersion}")
            self.log(f"   输入: {[input.name for input in spec.description.input]}")
            self.log(f"   输出: {[output.name for output in spec.description.output]}")
            
            # 2. 使用 YOLO 验证
            self.log("2. 使用 YOLO 验证...")
            model = YOLO(str(coreml_path))
            
            # 3. 测试推理
            self.log("3. 测试推理...")
            test_image_path = self.get_test_image()
            
            results = model.predict(test_image_path, verbose=False)
            
            if results:
                result = results[0]
                detections = len(result.boxes) if result.boxes is not None else 0
                self.log(f"   ✅ 推理成功，检测到 {detections} 个对象")
                
                # 保存验证结果
                save_dir = Path("02_coreml_conversion") / "validation_results"
                save_dir.mkdir(exist_ok=True)
                
                results = model.predict(
                    test_image_path, 
                    save=True, 
                    project=str(save_dir),
                    name=f"test_{coreml_path.stem}",
                    verbose=False
                )
                
                self.log(f"   验证结果已保存到: {save_dir}")
                return True
            else:
                self.log("   ❌ 推理失败", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ 模型验证失败: {e}", "ERROR")
            return False
    
    def get_test_image(self):
        """获取测试图片"""
        test_image_path = self.project_root / "shared_resources" / "test_images" / "bus.jpg"
        
        if test_image_path.exists():
            return str(test_image_path)
        
        # 如果本地没有，下载一个
        self.log("下载测试图片...")
        try:
            url = "https://ultralytics.com/images/bus.jpg"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            test_image_path.parent.mkdir(parents=True, exist_ok=True)
            with open(test_image_path, 'wb') as f:
                f.write(response.content)
            
            return str(test_image_path)
        except:
            return "https://ultralytics.com/images/bus.jpg"
    
    def compare_models(self, pt_model, coreml_path):
        """对比 PyTorch 和 CoreML 模型性能"""
        self.log("对比 PyTorch 和 CoreML 模型...")
        
        test_image = self.get_test_image()
        
        # PyTorch 模型测试
        self.log("测试 PyTorch 模型性能...")
        pt_times = []
        for i in range(10):
            start_time = time.time()
            pt_model.predict(test_image, verbose=False)
            pt_times.append(time.time() - start_time)
        
        pt_avg_time = np.mean(pt_times)
        
        # CoreML 模型测试
        self.log("测试 CoreML 模型性能...")
        coreml_model = YOLO(str(coreml_path))
        coreml_times = []
        for i in range(10):
            start_time = time.time()
            coreml_model.predict(test_image, verbose=False)
            coreml_times.append(time.time() - start_time)
        
        coreml_avg_time = np.mean(coreml_times)
        
        # 输出对比结果
        self.log("📊 性能对比结果:")
        self.log(f"   PyTorch 模型: {pt_avg_time:.3f}s ({1/pt_avg_time:.1f} FPS)")
        self.log(f"   CoreML 模型: {coreml_avg_time:.3f}s ({1/coreml_avg_time:.1f} FPS)")
        
        if coreml_avg_time < pt_avg_time:
            improvement = (pt_avg_time - coreml_avg_time) / pt_avg_time * 100
            self.log(f"   🚀 CoreML 模型快 {improvement:.1f}%")
        else:
            degradation = (coreml_avg_time - pt_avg_time) / pt_avg_time * 100
            self.log(f"   ⚠️  CoreML 模型慢 {degradation:.1f}%")
    
    def run_full_conversion(self):
        """运行完整转换流程"""
        self.log("🚀 开始 CoreML 转换和验证流程")
        self.log("=" * 60)
        
        # 1. 检查 PyTorch 模型
        result = self.check_pytorch_model()
        if result is None:
            return False
        
        pt_model, pt_model_path = result
        
        # 2. 转换为 CoreML
        coreml_path = self.convert_to_coreml(pt_model, pt_model_path)
        if coreml_path is None:
            return False
        
        # 3. 验证基础模型
        if not self.validate_coreml_model(coreml_path):
            return False
        
        # 4. 优化模型
        optimized_path = self.optimize_coreml_model(coreml_path)
        if optimized_path:
            # 验证优化后的模型
            if self.validate_coreml_model(optimized_path):
                self.log("✅ 优化模型验证通过")
            else:
                self.log("⚠️  优化模型验证失败，但基础模型可用", "WARNING")
        
        # 5. 性能对比
        self.compare_models(pt_model, coreml_path)
        
        # 6. 输出总结
        self.log("=" * 60)
        self.log("🎉 CoreML 转换和验证完成！")
        self.log(f"📁 输出目录: {self.output_dir}")
        self.log("📋 生成的文件:")
        
        for file_path in self.output_dir.glob("*"):
            if file_path.is_dir():
                size = sum(f.stat().st_size for f in file_path.rglob('*') if f.is_file())
                size_mb = size / 1024 / 1024
                self.log(f"   - {file_path.name} ({size_mb:.1f} MB)")
        
        self.log("\n📋 下一步: 运行 03_python_sdk 创建 Python SDK")
        
        return True

def main():
    """主函数"""
    converter = CoreMLConverter()
    success = converter.run_full_conversion()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)