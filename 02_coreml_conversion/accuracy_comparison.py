#!/usr/bin/env python3
"""
模型精度对比工具
比较 PyTorch 原始模型和 CoreML 转换模型的检测精度
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class AccuracyComparator:
    def __init__(self, pytorch_model_path: str, coreml_model_path: str):
        """
        初始化精度对比器
        
        Args:
            pytorch_model_path: PyTorch 模型路径
            coreml_model_path: CoreML 模型路径
        """
        self.pytorch_model_path = Path(pytorch_model_path)
        self.coreml_model_path = Path(coreml_model_path)
        
        # 加载模型
        self.pytorch_model = None
        self.coreml_model = None
        self.load_models()
        
        # 测试结果存储
        self.results = {
            'pytorch': [],
            'coreml': [],
            'comparison': {}
        }
    
    def load_models(self):
        """加载两个模型"""
        try:
            print("🔄 加载 PyTorch 模型...")
            self.pytorch_model = YOLO(str(self.pytorch_model_path))
            print("✅ PyTorch 模型加载成功")
            
            print("🔄 加载 CoreML 模型...")
            self.coreml_model = YOLO(str(self.coreml_model_path))
            print("✅ CoreML 模型加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)
    
    def prepare_test_dataset(self, dataset_name: str = "auto") -> List[str]:
        """准备测试数据集"""
        print("📥 准备测试数据集...")
        
        # 导入数据集管理器
        from dataset_manager import DatasetManager
        
        dataset_manager = DatasetManager()
        
        if dataset_name == "auto":
            # 自动选择最佳可用数据集
            print("🔍 自动选择测试数据集...")
            
            # 优先级顺序：COCO样本 > 自定义样本 > YOLO测试图片
            preferred_datasets = ["coco_val_sample", "custom_sample", "yolo_test_images"]
            
            for dataset in preferred_datasets:
                images = dataset_manager.get_dataset_images(dataset)
                if images:
                    print(f"✅ 使用现有数据集: {dataset} ({len(images)} 张图片)")
                    return images
            
            # 如果没有现有数据集，下载默认数据集
            print("📥 下载默认测试数据集...")
            dataset_path = dataset_manager.download_dataset("coco_val_sample")
            if dataset_path:
                images = dataset_manager.get_dataset_images("coco_val_sample")
                print(f"✅ 使用新下载的数据集: coco_val_sample ({len(images)} 张图片)")
                return images
            
            # 备选方案：下载 YOLO 测试图片
            print("📥 下载 YOLO 测试图片作为备选...")
            dataset_path = dataset_manager.download_dataset("yolo_test_images")
            if dataset_path:
                images = dataset_manager.get_dataset_images("yolo_test_images")
                return images
        
        else:
            # 使用指定的数据集
            images = dataset_manager.get_dataset_images(dataset_name)
            if images:
                print(f"✅ 使用指定数据集: {dataset_name} ({len(images)} 张图片)")
                return images
            
            # 尝试下载指定数据集
            print(f"📥 下载数据集: {dataset_name}")
            dataset_path = dataset_manager.download_dataset(dataset_name)
            if dataset_path:
                images = dataset_manager.get_dataset_images(dataset_name)
                return images
        
        # 最后的备选方案：使用本地图片
        print("⚠️  使用本地备选图片...")
        fallback_images = []
        local_test_image = project_root / "shared_resources" / "test_images" / "bus.jpg"
        if local_test_image.exists():
            fallback_images.append(str(local_test_image))
        
        if not fallback_images:
            print("❌ 无法获取测试图片")
            return []
        
        return fallback_images
    
    def run_inference(self, model: YOLO, image_path: str, conf_threshold: float = 0.25) -> List[Dict]:
        """
        运行推理并返回标准化结果
        
        Args:
            model: YOLO 模型
            image_path: 图片路径
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果列表
        """
        try:
            results = model.predict(image_path, conf=conf_threshold, verbose=False)
            
            detections = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    detection = {
                        'class_id': int(box.cls.cpu().numpy()[0]),
                        'class_name': model.names[int(box.cls.cpu().numpy()[0])],
                        'confidence': float(box.conf.cpu().numpy()[0]),
                        'bbox': box.xyxy.cpu().numpy()[0].tolist(),  # [x1, y1, x2, y2]
                        'area': float((box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return []
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        计算两个边界框的 IoU (Intersection over Union)
        
        Args:
            box1, box2: [x1, y1, x2, y2] 格式的边界框
            
        Returns:
            IoU 值
        """
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算并集区域
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections(self, pytorch_dets: List[Dict], coreml_dets: List[Dict], 
                        iou_threshold: float = 0.5) -> Tuple[List, List, List]:
        """
        匹配两个模型的检测结果
        
        Args:
            pytorch_dets: PyTorch 模型检测结果
            coreml_dets: CoreML 模型检测结果  
            iou_threshold: IoU 匹配阈值
            
        Returns:
            (matched_pairs, unmatched_pytorch, unmatched_coreml)
        """
        matched_pairs = []
        unmatched_pytorch = list(range(len(pytorch_dets)))
        unmatched_coreml = list(range(len(coreml_dets)))
        
        # 计算所有检测框之间的 IoU
        for i, pt_det in enumerate(pytorch_dets):
            best_iou = 0
            best_match = -1
            
            for j, cm_det in enumerate(coreml_dets):
                if j not in unmatched_coreml:
                    continue
                    
                # 只匹配相同类别的检测
                if pt_det['class_id'] != cm_det['class_id']:
                    continue
                
                iou = self.calculate_iou(pt_det['bbox'], cm_det['bbox'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match = j
            
            if best_match != -1:
                matched_pairs.append({
                    'pytorch_idx': i,
                    'coreml_idx': best_match,
                    'iou': best_iou,
                    'pytorch_det': pt_det,
                    'coreml_det': coreml_dets[best_match],
                    'conf_diff': abs(pt_det['confidence'] - coreml_dets[best_match]['confidence'])
                })
                unmatched_pytorch.remove(i)
                unmatched_coreml.remove(best_match)
        
        return matched_pairs, unmatched_pytorch, unmatched_coreml
    
    def calculate_ap_for_class(self, all_detections: List[Dict], class_id: int, 
                              iou_threshold: float = 0.5) -> float:
        """
        计算特定类别的 Average Precision (AP)
        
        Args:
            all_detections: 所有检测结果，包含 pytorch 和 coreml 的匹配信息
            class_id: 类别 ID
            iou_threshold: IoU 阈值
            
        Returns:
            该类别的 AP 值
        """
        # 收集该类别的所有检测
        class_detections = []
        
        for detection_data in all_detections:
            pytorch_dets = detection_data['pytorch_details']
            coreml_dets = detection_data['coreml_details'] 
            matched_pairs = detection_data['matched_details']
            
            # 获取该类别的 PyTorch 真值检测（作为 Ground Truth）
            gt_boxes = [det for det in pytorch_dets if det['class_id'] == class_id]
            # 获取该类别的 CoreML 预测检测
            pred_boxes = [det for det in coreml_dets if det['class_id'] == class_id]
            
            if not gt_boxes and not pred_boxes:
                continue
            
            # 计算该图片该类别的检测结果
            for pred in pred_boxes:
                # 找到最匹配的真值框
                best_iou = 0
                is_true_positive = False
                
                for gt in gt_boxes:
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        if iou >= iou_threshold:
                            is_true_positive = True
                
                class_detections.append({
                    'confidence': pred['confidence'],
                    'is_tp': is_true_positive,
                    'iou': best_iou
                })
        
        if not class_detections:
            return 0.0
        
        # 按置信度排序
        class_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 计算 Precision-Recall 曲线
        tp_count = 0
        total_gt = sum(len([det for det in data['pytorch_details'] if det['class_id'] == class_id]) 
                      for data in all_detections)
        
        if total_gt == 0:
            return 0.0
        
        precisions = []
        recalls = []
        
        for i, det in enumerate(class_detections):
            if det['is_tp']:
                tp_count += 1
            
            precision = tp_count / (i + 1)
            recall = tp_count / total_gt
            
            precisions.append(precision)
            recalls.append(recall)
        
        # 计算 AP (使用梯形规则积分)
        if len(recalls) < 2:
            return precisions[0] if precisions else 0.0
        
        # 11点插值方法计算 AP
        ap = 0.0
        for recall_threshold in np.arange(0, 1.1, 0.1):
            # 找到 recall >= threshold 的最大 precision
            max_precision = 0.0
            for p, r in zip(precisions, recalls):
                if r >= recall_threshold:
                    max_precision = max(max_precision, p)
            ap += max_precision
        
        return ap / 11.0
    
    def calculate_map(self, all_results: List[Dict], iou_threshold: float = 0.5) -> Dict:
        """
        计算 mAP (mean Average Precision)
        
        Args:
            all_results: 所有图片的检测结果
            iou_threshold: IoU 阈值
            
        Returns:
            包含 mAP 和各类别 AP 的字典
        """
        print(f"📊 计算 mAP (IoU={iou_threshold})...")
        
        # 获取所有出现的类别
        all_classes = set()
        for result in all_results:
            for det in result['pytorch_details']:
                all_classes.add(det['class_id'])
            for det in result['coreml_details']:
                all_classes.add(det['class_id'])
        
        if not all_classes:
            return {'mAP': 0.0, 'class_APs': {}}
        
        # 计算每个类别的 AP
        class_aps = {}
        valid_aps = []
        
        for class_id in sorted(all_classes):
            ap = self.calculate_ap_for_class(all_results, class_id, iou_threshold)
            
            # 获取类别名称
            class_name = "unknown"
            for result in all_results:
                for det in result['pytorch_details'] + result['coreml_details']:
                    if det['class_id'] == class_id:
                        class_name = det['class_name']
                        break
                if class_name != "unknown":
                    break
            
            class_aps[class_name] = ap
            if ap > 0:  # 只计算有效的 AP 值
                valid_aps.append(ap)
            
            print(f"   {class_name} (ID:{class_id}): AP = {ap:.3f}")
        
        # 计算 mAP
        map_score = np.mean(valid_aps) if valid_aps else 0.0
        
        return {
            'mAP': map_score,
            'class_APs': class_aps,
            'num_classes': len(all_classes),
            'valid_classes': len(valid_aps)
        }
    
    def analyze_single_image(self, image_path: str, conf_threshold: float = 0.25) -> Dict:
        """
        分析单张图片的精度对比
        """
        print(f"🔍 分析图片: {Path(image_path).name}")
        
        # 运行两个模型的推理
        pytorch_dets = self.run_inference(self.pytorch_model, image_path, conf_threshold)
        coreml_dets = self.run_inference(self.coreml_model, image_path, conf_threshold)
        
        # 匹配检测结果
        matched_pairs, unmatched_pt, unmatched_cm = self.match_detections(pytorch_dets, coreml_dets)
        
        # 计算统计信息
        analysis = {
            'image_path': image_path,
            'pytorch_detections': len(pytorch_dets),
            'coreml_detections': len(coreml_dets),
            'matched_pairs': len(matched_pairs),
            'unmatched_pytorch': len(unmatched_pt),
            'unmatched_coreml': len(unmatched_cm),
            'precision': len(matched_pairs) / len(coreml_dets) if len(coreml_dets) > 0 else 0,
            'recall': len(matched_pairs) / len(pytorch_dets) if len(pytorch_dets) > 0 else 0,
            'matched_details': matched_pairs,
            'pytorch_details': pytorch_dets,
            'coreml_details': coreml_dets
        }
        
        # 计算 F1 分数
        if analysis['precision'] + analysis['recall'] > 0:
            analysis['f1_score'] = 2 * (analysis['precision'] * analysis['recall']) / (analysis['precision'] + analysis['recall'])
        else:
            analysis['f1_score'] = 0
        
        # 计算平均置信度差异
        if matched_pairs:
            analysis['avg_confidence_diff'] = np.mean([pair['conf_diff'] for pair in matched_pairs])
            analysis['avg_iou'] = np.mean([pair['iou'] for pair in matched_pairs])
        else:
            analysis['avg_confidence_diff'] = 0
            analysis['avg_iou'] = 0
        
        print(f"   PyTorch: {len(pytorch_dets)} 检测, CoreML: {len(coreml_dets)} 检测")
        print(f"   匹配: {len(matched_pairs)}, 精确度: {analysis['precision']:.3f}, 召回率: {analysis['recall']:.3f}")
        
        return analysis
    
    def run_full_comparison(self, conf_thresholds: List[float] = [0.1, 0.25, 0.5], 
                           dataset_name: str = "auto") -> Dict:
        """
        运行完整的精度对比分析
        
        Args:
            conf_thresholds: 置信度阈值列表
            dataset_name: 测试数据集名称 ("auto", "coco_val_sample", "custom_sample", "yolo_test_images")
        """
        print("🚀 开始模型精度对比分析")
        print("=" * 60)
        
        # 获取测试数据集
        test_images = self.prepare_test_dataset(dataset_name)
        if not test_images:
            print("❌ 没有可用的测试图片")
            return {}
        
        print(f"📊 将在 {len(test_images)} 张图片上进行精度测试")
        
        # 对每个置信度阈值进行测试
        all_results = {}
        
        for conf_thresh in conf_thresholds:
            print(f"\n📊 测试置信度阈值: {conf_thresh}")
            print("-" * 40)
            
            threshold_results = []
            
            for i, image_path in enumerate(test_images):
                try:
                    print(f"   处理图片 {i+1}/{len(test_images)}: {Path(image_path).name}")
                    analysis = self.analyze_single_image(image_path, conf_thresh)
                    threshold_results.append(analysis)
                except Exception as e:
                    print(f"   ⚠️  分析失败: {Path(image_path).name} - {e}")
                    continue
            
            # 计算总体统计
            if threshold_results:
                overall_stats = self.calculate_overall_stats(threshold_results)
                
                # 计算 mAP
                map_results = self.calculate_map(threshold_results, iou_threshold=0.5)
                
                all_results[conf_thresh] = {
                    'individual_results': threshold_results,
                    'overall_stats': overall_stats,
                    'map_results': map_results
                }
                
                print(f"\n📈 置信度 {conf_thresh} 总体统计:")
                print(f"   平均精确度: {overall_stats['avg_precision']:.3f}")
                print(f"   平均召回率: {overall_stats['avg_recall']:.3f}")
                print(f"   平均 F1 分数: {overall_stats['avg_f1']:.3f}")
                print(f"   mAP@0.5: {map_results['mAP']:.3f}")
                print(f"   平均置信度差异: {overall_stats['avg_conf_diff']:.3f}")
                print(f"   平均 IoU: {overall_stats['avg_iou']:.3f}")
                
                if len(test_images) >= 10:
                    print(f"   测试图片数量: {len(test_images)} (充足)")
                else:
                    print(f"   测试图片数量: {len(test_images)} (建议增加)")
        
        # 保存结果
        self.save_results(all_results)
        
        # 生成分析报告
        self.generate_report(all_results)
        
        return all_results
    
    def calculate_overall_stats(self, results: List[Dict]) -> Dict:
        """计算总体统计信息"""
        if not results:
            return {}
        
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        conf_diffs = [r['avg_confidence_diff'] for r in results]
        ious = [r['avg_iou'] for r in results]
        
        return {
            'avg_precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'avg_recall': np.mean(recalls),
            'std_recall': np.std(recalls),
            'avg_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'avg_conf_diff': np.mean(conf_diffs),
            'std_conf_diff': np.std(conf_diffs),
            'avg_iou': np.mean(ious),
            'std_iou': np.std(ious),
            'total_images': len(results)
        }
    
    def save_results(self, results: Dict):
        """保存结果到JSON文件"""
        output_dir = Path("02_coreml_conversion") / "accuracy_results"
        output_dir.mkdir(exist_ok=True)
        
        # 保存详细结果
        results_file = output_dir / f"accuracy_comparison_{int(time.time())}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 详细结果已保存: {results_file}")
    
    def generate_report(self, results: Dict):
        """生成精度分析报告"""
        if not results:
            return
        
        output_dir = Path("02_coreml_conversion") / "accuracy_results"
        output_dir.mkdir(exist_ok=True)
        
        # 生成文本报告
        report_file = output_dir / f"accuracy_report_{int(time.time())}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# YOLOv11 CoreML 精度对比报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"PyTorch 模型: {self.pytorch_model_path.name}\n")
            f.write(f"CoreML 模型: {self.coreml_model_path.name}\n\n")
            
            # 总体结论
            f.write("## 📊 总体结论\n\n")
            
            best_f1_thresh = max(results.keys(), key=lambda k: results[k]['overall_stats']['avg_f1'])
            best_stats = results[best_f1_thresh]['overall_stats']
            best_map = results[best_f1_thresh]['map_results']
            
            f.write(f"- **最佳置信度阈值**: {best_f1_thresh}\n")
            f.write(f"- **平均精确度**: {best_stats['avg_precision']:.3f} ± {best_stats['std_precision']:.3f}\n")
            f.write(f"- **平均召回率**: {best_stats['avg_recall']:.3f} ± {best_stats['std_recall']:.3f}\n")
            f.write(f"- **平均 F1 分数**: {best_stats['avg_f1']:.3f} ± {best_stats['std_f1']:.3f}\n")
            f.write(f"- **mAP@0.5**: {best_map['mAP']:.3f} (基于 {best_map['valid_classes']}/{best_map['num_classes']} 个类别)\n")
            f.write(f"- **平均置信度差异**: {best_stats['avg_conf_diff']:.3f} ± {best_stats['std_conf_diff']:.3f}\n")
            f.write(f"- **平均 IoU**: {best_stats['avg_iou']:.3f} ± {best_stats['std_iou']:.3f}\n")
            f.write(f"- **测试图片数量**: {best_stats['total_images']} 张\n\n")
            
            # 精度评估
            if best_stats['avg_f1'] > 0.9:
                f.write("✅ **精度评估**: 优秀 - CoreML 转换基本无精度损失\n\n")
            elif best_stats['avg_f1'] > 0.8:
                f.write("🟡 **精度评估**: 良好 - CoreML 转换有轻微精度损失\n\n") 
            elif best_stats['avg_f1'] > 0.7:
                f.write("🟠 **精度评估**: 一般 - CoreML 转换有明显精度损失\n\n")
            else:
                f.write("❌ **精度评估**: 较差 - CoreML 转换精度损失严重\n\n")
            
            # 详细统计
            f.write("## 📈 详细统计\n\n")
            f.write("| 置信度阈值 | 精确度 | 召回率 | F1分数 | 置信度差异 | 平均IoU |\n")
            f.write("|------------|---------|---------|---------|------------|----------|\n")
            
            for thresh, data in results.items():
                stats = data['overall_stats']
                f.write(f"| {thresh} | {stats['avg_precision']:.3f} | {stats['avg_recall']:.3f} | "
                       f"{stats['avg_f1']:.3f} | {stats['avg_conf_diff']:.3f} | {stats['avg_iou']:.3f} |\n")
            
            f.write("\n## 🔍 分析建议\n\n")
            
            if best_stats['avg_conf_diff'] > 0.1:
                f.write("- ⚠️  置信度差异较大，建议检查模型转换参数\n")
            
            if best_stats['avg_iou'] < 0.8:
                f.write("- ⚠️  IoU 较低，检测框位置可能有偏差\n")
            
            if best_stats['avg_precision'] < 0.8:
                f.write("- ⚠️  精确度偏低，可能存在较多误检\n")
            
            if best_stats['avg_recall'] < 0.8:
                f.write("- ⚠️  召回率偏低，可能遗漏较多对象\n")
        
        print(f"📄 精度报告已生成: {report_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv11 CoreML 精度对比工具")
    parser.add_argument("--pytorch-model", required=True, help="PyTorch 模型路径")
    parser.add_argument("--coreml-model", required=True, help="CoreML 模型路径")
    parser.add_argument("--conf-thresholds", nargs="+", type=float, 
                       default=[0.1, 0.25, 0.5], help="置信度阈值列表")
    parser.add_argument("--dataset", type=str, default="auto",
                       choices=["auto", "coco_val_sample", "custom_sample", "yolo_test_images"],
                       help="测试数据集选择")
    parser.add_argument("--download-datasets", action="store_true", 
                       help="预先下载所有测试数据集")
    
    args = parser.parse_args()
    
    # 如果只是下载数据集
    if args.download_datasets:
        from dataset_manager import DatasetManager
        manager = DatasetManager()
        manager.download_all_datasets()
        return
    
    # 检查模型文件存在性
    if not Path(args.pytorch_model).exists():
        print(f"❌ PyTorch 模型文件不存在: {args.pytorch_model}")
        sys.exit(1)
    
    if not Path(args.coreml_model).exists():
        print(f"❌ CoreML 模型文件不存在: {args.coreml_model}")
        sys.exit(1)
    
    # 运行精度对比
    comparator = AccuracyComparator(args.pytorch_model, args.coreml_model)
    results = comparator.run_full_comparison(args.conf_thresholds, args.dataset)
    
    if results:
        print("\n🎉 精度对比分析完成！")
        print("\n查看详细报告:")
        print(f"   - 结果文件: 02_coreml_conversion/accuracy_results/")
        
        # 显示最佳结果摘要
        best_thresh = max(results.keys(), key=lambda k: results[k]['overall_stats']['avg_f1'])
        best_results = results[best_thresh]
        
        print(f"\n📊 最佳结果摘要 (置信度阈值: {best_thresh}):")
        print(f"   F1 分数: {best_results['overall_stats']['avg_f1']:.3f}")
        print(f"   mAP@0.5: {best_results['map_results']['mAP']:.3f}")
        print(f"   测试图片: {best_results['overall_stats']['total_images']} 张")
        
        if best_results['overall_stats']['avg_f1'] > 0.9:
            print("   🎯 精度评估: 优秀 - 转换质量很高")
        elif best_results['overall_stats']['avg_f1'] > 0.8:
            print("   🟡 精度评估: 良好 - 转换质量可接受")
        else:
            print("   ⚠️  精度评估: 需要优化 - 建议检查转换参数")
            
    else:
        print("❌ 精度对比分析失败")
        sys.exit(1)

if __name__ == "__main__":
    main()