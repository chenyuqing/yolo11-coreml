"""
YOLOv11 CoreML Python SDK

这个 SDK 提供了一个简单易用的接口来使用 YOLOv11 CoreML 模型进行对象检测。
"""

import os
import sys
from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

__version__ = "1.0.0"
__author__ = "YOLOv11 CoreML Team"

class YOLOv11CoreML:
    """
    YOLOv11 CoreML 推理类
    
    这个类封装了 YOLOv11 CoreML 模型，提供简单的预测接口。
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化模型
        
        Args:
            model_path: CoreML 模型路径。如果为 None，会自动查找默认模型。
        """
        self.model_path = self._resolve_model_path(model_path)
        self.model = self._load_model()
        self._class_names = None
    
    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """解析模型路径"""
        if model_path is not None:
            if os.path.exists(model_path):
                return model_path
            else:
                raise FileNotFoundError(f"指定的模型文件不存在: {model_path}")
        
        # 自动查找模型
        search_paths = [
            # 相对于当前文件的路径
            Path(__file__).parent / ".." / ".." / "models" / "yolo11n.mlpackage",
            # 项目根目录
            Path(__file__).parent / ".." / ".." / ".." / "shared_resources" / "models" / "yolo11n.mlpackage",
            Path(__file__).parent / ".." / ".." / ".." / "02_coreml_conversion" / "coreml_models" / "yolo11n.mlpackage",
            # 当前工作目录
            Path.cwd() / "yolo11n.mlpackage",
            # 打包的模型
            Path(__file__).parent / "models" / "yolo11n.mlpackage",
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path.resolve())
        
        raise FileNotFoundError(
            f"找不到 CoreML 模型文件。请确保模型文件存在于以下位置之一:\\n" +
            "\\n".join([f"  - {path}" for path in search_paths]) +
            "\\n\\n或者在初始化时指定模型路径: YOLOv11CoreML(model_path='your_model.mlpackage')"
        )
    
    def _load_model(self) -> YOLO:
        """加载模型"""
        try:
            model = YOLO(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    @property
    def class_names(self) -> dict:
        """获取类别名称映射"""
        if self._class_names is None:
            self._class_names = self.model.names
        return self._class_names
    
    def predict(
        self, 
        source: Union[str, np.ndarray, Path], 
        conf: float = 0.25,
        iou: float = 0.7,
        save: bool = False,
        save_dir: str = "runs/detect/predict",
        show_labels: bool = True,
        show_conf: bool = True,
        line_width: int = 2
    ) -> List[Results]:
        """
        执行目标检测预测
        
        Args:
            source: 输入源（图片路径、URL、numpy数组等）
            conf: 置信度阈值
            iou: IoU 阈值用于 NMS
            save: 是否保存结果图片
            save_dir: 保存目录
            show_labels: 是否显示类别标签
            show_conf: 是否显示置信度
            line_width: 边框线宽
            
        Returns:
            检测结果列表
        """
        try:
            results = self.model.predict(
                source=source,
                conf=conf,
                iou=iou,
                save=save,
                project=save_dir,
                show_labels=show_labels,
                show_conf=show_conf,
                line_width=line_width,
                verbose=False
            )
            return results
        except Exception as e:
            raise RuntimeError(f"预测失败: {e}")
    
    def predict_and_parse(
        self, 
        source: Union[str, np.ndarray, Path],
        conf: float = 0.25,
        iou: float = 0.7
    ) -> List[dict]:
        """
        执行预测并返回解析后的结果
        
        Args:
            source: 输入源
            conf: 置信度阈值
            iou: IoU 阈值
            
        Returns:
            解析后的检测结果列表，每个元素包含:
            - class_id: 类别ID
            - class_name: 类别名称
            - confidence: 置信度
            - bbox: 边界框 [x1, y1, x2, y2]
        """
        results = self.predict(source, conf=conf, iou=iou, save=False)
        
        parsed_results = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.cpu().numpy()[0])
                    confidence = float(box.conf.cpu().numpy()[0])
                    bbox = box.xyxy.cpu().numpy()[0].tolist()  # [x1, y1, x2, y2]
                    
                    parsed_results.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'confidence': confidence,
                        'bbox': bbox
                    })
        
        return parsed_results
    
    def info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_path': self.model_path,
            'model_type': 'YOLOv11 CoreML',
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'sdk_version': __version__
        }
    
    def benchmark(
        self, 
        source: Union[str, np.ndarray, Path],
        num_runs: int = 10,
        warmup_runs: int = 3
    ) -> dict:
        """
        性能基准测试
        
        Args:
            source: 测试输入源
            num_runs: 测试运行次数
            warmup_runs: 预热运行次数
            
        Returns:
            性能统计信息
        """
        import time
        
        # 预热
        for _ in range(warmup_runs):
            self.predict(source, save=False)
        
        # 基准测试
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.predict(source, save=False)
            end_time = time.time()
            times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            'num_runs': num_runs,
            'avg_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'avg_fps': float(1.0 / np.mean(times)),
            'median_time': float(np.median(times))
        }

# 向后兼容的别名
YOLOv11CoreMLPredictor = YOLOv11CoreML

# 导出的公共接口
__all__ = [
    'YOLOv11CoreML',
    'YOLOv11CoreMLPredictor',
    '__version__',
    '__author__'
]