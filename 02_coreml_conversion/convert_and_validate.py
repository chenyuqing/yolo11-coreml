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
        self.output_dir = Path(__file__).parent / "coreml_models"
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
            # 先转换不带 NMS 的版本（用于精度测试）
            self.log("1. 转换不带 NMS 的版本（用于精度测试）...")
            model.export(
                format='coreml',
                optimize=True,
                nms=False
            )
            
            # 处理不带 NMS 的文件
            model_name = pt_model_path.stem
            mlpackage_name_no_nms = f"{model_name}.mlpackage"
            source_path_no_nms = pt_model_path.parent / mlpackage_name_no_nms
            
            if source_path_no_nms.exists():
                dest_path_no_nms = self.output_dir / f"{model_name}_no_nms.mlpackage"
                if dest_path_no_nms.exists():
                    shutil.rmtree(dest_path_no_nms)
                shutil.move(str(source_path_no_nms), dest_path_no_nms)
                self.log(f"✅ 无 NMS 版本转换完成: {dest_path_no_nms}")
            
            # 再转换带 NMS 的版本（用于部署）
            self.log("2. 转换带 NMS 的版本（用于部署）...")
            model.export(
                format='coreml',
                optimize=True,
                nms=True
            )
            
            # 处理带 NMS 的文件
            mlpackage_name_with_nms = f"{model_name}.mlpackage"
            source_path_with_nms = pt_model_path.parent / mlpackage_name_with_nms
            
            if source_path_with_nms.exists():
                dest_path_with_nms = self.output_dir / f"{model_name}_with_nms.mlpackage"
                if dest_path_with_nms.exists():
                    shutil.rmtree(dest_path_with_nms)
                shutil.move(str(source_path_with_nms), dest_path_with_nms)
                self.log(f"✅ 带 NMS 版本转换完成: {dest_path_with_nms}")
            
            # 返回不带 NMS 的版本用于验证和测试
            no_nms_path = self.output_dir / f"{model_name}_no_nms.mlpackage"
            if no_nms_path.exists():
                self.log(f"✅ CoreML 模型转换完成，返回无 NMS 版本进行验证")
                return no_nms_path
            else:
                self.log(f"❌ 转换失败，未找到无 NMS 模型", "ERROR")
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
                save_dir = Path(__file__).parent / "validation_results"
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
        """对比 PyTorch 和 CoreML 模型性能和精度"""
        self.log("对比 PyTorch 和 CoreML 模型...")
        
        test_image = self.get_test_image()
        
        # 性能对比
        self.log("📊 性能对比测试...")
        
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
        
        # 输出性能对比结果
        self.log("📊 性能对比结果:")
        self.log(f"   PyTorch 模型: {pt_avg_time:.3f}s ({1/pt_avg_time:.1f} FPS)")
        self.log(f"   CoreML 模型: {coreml_avg_time:.3f}s ({1/coreml_avg_time:.1f} FPS)")
        
        if coreml_avg_time < pt_avg_time:
            improvement = (pt_avg_time - coreml_avg_time) / pt_avg_time * 100
            self.log(f"   🚀 CoreML 模型快 {improvement:.1f}%")
        else:
            degradation = (coreml_avg_time - pt_avg_time) / pt_avg_time * 100
            self.log(f"   ⚠️  CoreML 模型慢 {degradation:.1f}%")
        
        # 精度对比
        self.log("🎯 精度对比测试...")
        self.compare_accuracy(pt_model, coreml_model, test_image)
    
    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1]) 
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections_by_iou(self, pt_detections, cm_detections, iou_thresh=0.5):
        """基于IoU匹配两个模型的检测结果"""
        matches = []
        used_cm_indices = set()
        
        for i, pt_det in enumerate(pt_detections):
            best_iou = 0
            best_match_idx = -1
            
            for j, cm_det in enumerate(cm_detections):
                if j in used_cm_indices:
                    continue
                
                # 只匹配相同类别
                if pt_det['class_id'] != cm_det['class_id']:
                    continue
                
                iou = self.calculate_iou(pt_det['bbox'], cm_det['bbox'])
                if iou > best_iou and iou >= iou_thresh:
                    best_iou = iou
                    best_match_idx = j
            
            if best_match_idx != -1:
                matches.append({
                    'pt_idx': i,
                    'cm_idx': best_match_idx,
                    'iou': best_iou,
                    'pt_det': pt_det,
                    'cm_det': cm_detections[best_match_idx],
                    'conf_diff': abs(pt_det['confidence'] - cm_detections[best_match_idx]['confidence']),
                    'bbox_shift': self.calculate_bbox_shift(pt_det['bbox'], cm_detections[best_match_idx]['bbox'])
                })
                used_cm_indices.add(best_match_idx)
        
        # 未匹配的检测
        unmatched_pt = [i for i in range(len(pt_detections)) if i not in [m['pt_idx'] for m in matches]]
        unmatched_cm = [i for i in range(len(cm_detections)) if i not in used_cm_indices]
        
        return matches, unmatched_pt, unmatched_cm
    
    def calculate_bbox_shift(self, bbox1, bbox2):
        """计算边界框的位置偏移"""
        # 计算中心点偏移
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        center_shift = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # 计算尺寸变化
        size1 = [(bbox1[2] - bbox1[0]), (bbox1[3] - bbox1[1])]
        size2 = [(bbox2[2] - bbox2[0]), (bbox2[3] - bbox2[1])]
        
        size_ratio = np.sqrt((size2[0] / size1[0])**2 + (size2[1] / size1[1])**2) if size1[0] > 0 and size1[1] > 0 else 1.0
        
        return {
            'center_shift': center_shift,
            'size_ratio': size_ratio
        }

    def compare_accuracy(self, pt_model, coreml_model, test_image):
        """对比两个模型的检测精度 - 改进版本"""
        try:
            # 使用相同的置信度阈值进行推理
            conf_threshold = 0.25
            
            # PyTorch 模型推理
            pt_results = pt_model.predict(test_image, conf=conf_threshold, verbose=False)
            pt_detections = []
            if pt_results and pt_results[0].boxes is not None:
                for box in pt_results[0].boxes:
                    pt_detections.append({
                        'class_id': int(box.cls.cpu().numpy()[0]),
                        'class_name': pt_model.names[int(box.cls.cpu().numpy()[0])],
                        'confidence': float(box.conf.cpu().numpy()[0]),
                        'bbox': box.xyxy.cpu().numpy()[0].tolist()
                    })
            
            # CoreML 模型推理
            cm_results = coreml_model.predict(test_image, conf=conf_threshold, verbose=False)
            cm_detections = []
            if cm_results and cm_results[0].boxes is not None:
                for box in cm_results[0].boxes:
                    cm_detections.append({
                        'class_id': int(box.cls.cpu().numpy()[0]),
                        'class_name': coreml_model.names[int(box.cls.cpu().numpy()[0])],
                        'confidence': float(box.conf.cpu().numpy()[0]),
                        'bbox': box.xyxy.cpu().numpy()[0].tolist()
                    })
            
            self.log(f"   PyTorch 检测数量: {len(pt_detections)}")
            self.log(f"   CoreML 检测数量: {len(cm_detections)}")
            
            if len(pt_detections) == 0 and len(cm_detections) == 0:
                self.log("   ✅ 两个模型都未检测到对象 - 结果一致")
                return
            
            if len(pt_detections) == 0 or len(cm_detections) == 0:
                self.log("   ⚠️  一个模型有检测，另一个没有 - 存在显著差异")
                return
            
            # 基于IoU进行精确匹配
            matches, unmatched_pt, unmatched_cm = self.match_detections_by_iou(pt_detections, cm_detections)
            
            self.log(f"   IoU匹配结果: {len(matches)}/{len(pt_detections)} 个检测被匹配")
            
            if len(matches) > 0:
                # 分析匹配的检测质量
                ious = [m['iou'] for m in matches]
                conf_diffs = [m['conf_diff'] for m in matches] 
                center_shifts = [m['bbox_shift']['center_shift'] for m in matches]
                size_ratios = [m['bbox_shift']['size_ratio'] for m in matches]
                
                self.log(f"   平均 IoU: {np.mean(ious):.3f} (±{np.std(ious):.3f})")
                self.log(f"   平均置信度差异: {np.mean(conf_diffs):.3f} (±{np.std(conf_diffs):.3f})")
                self.log(f"   平均中心偏移: {np.mean(center_shifts):.1f} 像素")
                self.log(f"   平均尺寸比例: {np.mean(size_ratios):.3f}")
                
                # 质量评估
                if np.mean(ious) > 0.8:
                    self.log("   ✅ 边界框精度: 优秀")
                elif np.mean(ious) > 0.6:
                    self.log("   🟡 边界框精度: 良好")
                else:
                    self.log("   ⚠️  边界框精度: 需要关注")
                
                if np.mean(conf_diffs) < 0.1:
                    self.log("   ✅ 置信度一致性: 优秀")
                elif np.mean(conf_diffs) < 0.2:
                    self.log("   🟡 置信度一致性: 良好")
                else:
                    self.log("   ⚠️  置信度一致性: 需要关注")
            
            # 分析未匹配的检测
            if unmatched_pt:
                self.log(f"   ⚠️  PyTorch 独有检测: {len(unmatched_pt)} 个")
                for idx in unmatched_pt[:3]:  # 只显示前3个
                    det = pt_detections[idx]
                    self.log(f"      - {det['class_name']} (置信度: {det['confidence']:.2f})")
            
            if unmatched_cm:
                self.log(f"   ⚠️  CoreML 独有检测: {len(unmatched_cm)} 个") 
                for idx in unmatched_cm[:3]:  # 只显示前3个
                    det = cm_detections[idx]
                    self.log(f"      - {det['class_name']} (置信度: {det['confidence']:.2f})")
            
            # 计算总体精度指标
            precision = len(matches) / len(cm_detections) if len(cm_detections) > 0 else 0
            recall = len(matches) / len(pt_detections) if len(pt_detections) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            self.log(f"   精确度: {precision:.3f}, 召回率: {recall:.3f}, F1: {f1:.3f}")
            
            # 总体评估
            if f1 > 0.9 and len(matches) > 0:
                self.log("   🎯 总体评估: 优秀 - CoreML转换质量很高")
            elif f1 > 0.7:
                self.log("   🟡 总体评估: 良好 - CoreML转换质量可接受")
            elif f1 > 0.5:
                self.log("   🟠 总体评估: 一般 - 建议检查转换参数")
            else:
                self.log("   ❌ 总体评估: 较差 - 存在显著精度损失")
            
            # 详细精度分析建议
            self.log("\n💡 详细精度分析:")
            self.log("   运行完整测试集精度分析：")
            pt_model_path = self.project_root / "shared_resources" / "models" / "yolo11n.pt"
            coreml_model_path = self.output_dir / "yolo11n.mlpackage"
            self.log(f"   python accuracy_comparison.py --pytorch-model {pt_model_path} --coreml-model {coreml_model_path}")
            
        except Exception as e:
            self.log(f"⚠️  精度对比失败: {e}")
            self.log("建议使用独立的精度分析工具进行详细对比")
    
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