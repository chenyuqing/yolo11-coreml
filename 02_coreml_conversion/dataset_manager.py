#!/usr/bin/env python3
"""
数据集管理工具
用于下载和管理精度测试数据集
"""

import os
import sys
import json
import zipfile
import tarfile
from pathlib import Path
from typing import List, Dict, Optional
import requests
from tqdm import tqdm
import yaml

class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, datasets_dir: str = "02_coreml_conversion/datasets"):
        """
        初始化数据集管理器
        
        Args:
            datasets_dir: 数据集存储目录
        """
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集配置
        self.available_datasets = {
            "coco_val_sample": {
                "name": "COCO 验证集样本",
                "description": "COCO 2017 验证集的 100 张图片样本",
                "size": "~20MB",
                "images": 100,
                "url": None,  # 我们将创建一个精选的样本集
                "format": "coco"
            },
            "yolo_test_images": {
                "name": "YOLO 官方测试图片",
                "description": "Ultralytics 官方提供的测试图片集",
                "size": "~5MB", 
                "images": 10,
                "urls": [
                    "https://ultralytics.com/images/bus.jpg",
                    "https://ultralytics.com/images/zidane.jpg",
                    "https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg",
                    "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg",
                    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
                    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg",
                ],
                "format": "images"
            },
            "custom_sample": {
                "name": "自定义测试样本",
                "description": "精选的多样化测试图片",
                "size": "~10MB",
                "images": 20,
                "format": "images"
            }
        }
    
    def download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """
        下载文件并显示进度条
        
        Args:
            url: 下载链接
            filepath: 保存路径
            chunk_size: 块大小
            
        Returns:
            是否下载成功
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            print(f"下载失败 {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def download_yolo_test_images(self) -> Path:
        """下载 YOLO 官方测试图片"""
        dataset_dir = self.datasets_dir / "yolo_test_images"
        dataset_dir.mkdir(exist_ok=True)
        
        print("📥 下载 YOLO 官方测试图片...")
        
        urls = self.available_datasets["yolo_test_images"]["urls"]
        successful_downloads = []
        
        for i, url in enumerate(urls):
            filename = f"test_image_{i+1:02d}.jpg"
            filepath = dataset_dir / filename
            
            if filepath.exists():
                print(f"   跳过已存在的文件: {filename}")
                successful_downloads.append(str(filepath))
                continue
            
            print(f"   下载: {filename}")
            if self.download_file(url, filepath):
                successful_downloads.append(str(filepath))
            else:
                print(f"   ⚠️  下载失败: {filename}")
        
        # 创建数据集信息文件
        dataset_info = {
            "name": "YOLO Test Images",
            "images": len(successful_downloads),
            "image_paths": successful_downloads,
            "format": "images",
            "created_at": str(Path().cwd())
        }
        
        info_file = dataset_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"✅ 成功下载 {len(successful_downloads)} 张图片到 {dataset_dir}")
        return dataset_dir
    
    def create_custom_sample_dataset(self) -> Path:
        """创建自定义样本数据集"""
        dataset_dir = self.datasets_dir / "custom_sample"
        dataset_dir.mkdir(exist_ok=True)
        
        print("📥 创建自定义测试样本...")
        
        # 多样化的测试图片 URLs
        sample_urls = [
            # 交通场景
            "https://ultralytics.com/images/bus.jpg",
            "https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg",
            
            # 人物场景  
            "https://ultralytics.com/images/zidane.jpg",
            "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg",
            
            # 来自 COCO 数据集的代表性图片
            "https://farm4.staticflickr.com/3273/2602356079_2e0cc0b89e_z.jpg",  # 猫
            "https://farm6.staticflickr.com/5285/5221301158_b396b8b03b_z.jpg",  # 狗
            "https://farm5.staticflickr.com/4062/4695028316_4b2d1d4502_z.jpg",  # 汽车
            "https://farm3.staticflickr.com/2534/4181635046_36ab77a4d5_z.jpg",  # 摩托车
            "https://farm6.staticflickr.com/5241/5245510697_e8e7e85636_z.jpg",  # 自行车
            "https://farm5.staticflickr.com/4148/4997038311_9de2e8c3b4_z.jpg",  # 飞机
        ]
        
        successful_downloads = []
        
        for i, url in enumerate(sample_urls):
            filename = f"sample_{i+1:02d}.jpg"
            filepath = dataset_dir / filename
            
            if filepath.exists():
                successful_downloads.append(str(filepath))
                continue
            
            print(f"   下载样本 {i+1}/{len(sample_urls)}: {filename}")
            if self.download_file(url, filepath):
                successful_downloads.append(str(filepath))
        
        # 创建数据集信息
        dataset_info = {
            "name": "Custom Sample Dataset",
            "description": "精选的多样化测试图片，包含不同场景和对象类型",
            "images": len(successful_downloads),
            "image_paths": successful_downloads,
            "categories": ["person", "vehicle", "animal", "outdoor", "indoor"],
            "format": "images"
        }
        
        info_file = dataset_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"✅ 自定义样本数据集创建完成: {len(successful_downloads)} 张图片")
        return dataset_dir
    
    def download_coco_val_sample(self) -> Path:
        """下载 COCO 验证集样本"""
        dataset_dir = self.datasets_dir / "coco_val_sample"
        dataset_dir.mkdir(exist_ok=True)
        
        print("📥 准备 COCO 验证集样本...")
        
        # 由于完整的 COCO 数据集很大，我们创建一个精选样本
        # 包含不同复杂度和对象数量的图片
        coco_sample_urls = [
            # 简单场景（1-2个对象）
            "http://images.cocodataset.org/val2017/000000000139.jpg",
            "http://images.cocodataset.org/val2017/000000000285.jpg",
            "http://images.cocodataset.org/val2017/000000000632.jpg",
            
            # 中等复杂度（3-5个对象）
            "http://images.cocodataset.org/val2017/000000000724.jpg", 
            "http://images.cocodataset.org/val2017/000000001268.jpg",
            "http://images.cocodataset.org/val2017/000000001503.jpg",
            
            # 复杂场景（6+个对象）
            "http://images.cocodataset.org/val2017/000000002299.jpg",
            "http://images.cocodataset.org/val2017/000000002532.jpg",
            "http://images.cocodataset.org/val2017/000000002685.jpg",
            
            # 不同类别
            "http://images.cocodataset.org/val2017/000000003156.jpg",  # 动物
            "http://images.cocodataset.org/val2017/000000003501.jpg",  # 食物
            "http://images.cocodataset.org/val2017/000000004395.jpg",  # 运动
            "http://images.cocodataset.org/val2017/000000005193.jpg",  # 室内
            "http://images.cocodataset.org/val2017/000000005529.jpg",  # 交通
            "http://images.cocodataset.org/val2017/000000006040.jpg",  # 人群
        ]
        
        print(f"下载 {len(coco_sample_urls)} 张 COCO 样本图片...")
        successful_downloads = []
        
        for i, url in enumerate(coco_sample_urls):
            # 从 URL 中提取图片 ID
            img_id = url.split('/')[-1]
            filepath = dataset_dir / img_id
            
            if filepath.exists():
                successful_downloads.append(str(filepath))
                continue
            
            print(f"   下载 COCO 图片 {i+1}/{len(coco_sample_urls)}: {img_id}")
            if self.download_file(url, filepath):
                successful_downloads.append(str(filepath))
        
        # 创建 COCO 格式的数据集信息
        dataset_info = {
            "name": "COCO Validation Sample",
            "description": "COCO 2017 验证集的精选样本，包含不同复杂度的场景",
            "images": len(successful_downloads),
            "image_paths": successful_downloads,
            "format": "coco",
            "categories": "COCO 80 classes",
            "complexity_levels": ["simple", "medium", "complex"],
            "source": "COCO 2017 validation set"
        }
        
        info_file = dataset_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"✅ COCO 样本数据集准备完成: {len(successful_downloads)} 张图片")
        return dataset_dir
    
    def list_available_datasets(self):
        """列出可用的数据集"""
        print("📋 可用的测试数据集:")
        print("=" * 60)
        
        for key, info in self.available_datasets.items():
            print(f"\n🔸 {key}")
            print(f"   名称: {info['name']}")
            print(f"   描述: {info['description']}")
            print(f"   大小: {info['size']}")
            print(f"   图片数量: {info['images']}")
            
            # 检查是否已下载
            dataset_path = self.datasets_dir / key
            if dataset_path.exists():
                info_file = dataset_path / "dataset_info.json"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        local_info = json.load(f)
                    print(f"   状态: ✅ 已下载 ({local_info['images']} 张图片)")
                else:
                    print(f"   状态: ⚠️  目录存在但信息不完整")
            else:
                print(f"   状态: ❌ 未下载")
    
    def download_dataset(self, dataset_name: str) -> Optional[Path]:
        """
        下载指定的数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            数据集路径，如果失败则返回 None
        """
        if dataset_name not in self.available_datasets:
            print(f"❌ 未知的数据集: {dataset_name}")
            return None
        
        print(f"开始下载数据集: {dataset_name}")
        
        if dataset_name == "yolo_test_images":
            return self.download_yolo_test_images()
        elif dataset_name == "custom_sample":
            return self.create_custom_sample_dataset()
        elif dataset_name == "coco_val_sample":
            return self.download_coco_val_sample()
        else:
            print(f"❌ 数据集 {dataset_name} 的下载方法未实现")
            return None
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """获取数据集信息"""
        dataset_path = self.datasets_dir / dataset_name
        info_file = dataset_path / "dataset_info.json"
        
        if not info_file.exists():
            return None
        
        with open(info_file, 'r') as f:
            return json.load(f)
    
    def get_dataset_images(self, dataset_name: str) -> List[str]:
        """获取数据集中的所有图片路径"""
        info = self.get_dataset_info(dataset_name)
        if info is None:
            return []
        
        return info.get('image_paths', [])
    
    def download_all_datasets(self):
        """下载所有可用的数据集"""
        print("🚀 下载所有测试数据集...")
        
        for dataset_name in self.available_datasets.keys():
            print(f"\n{'='*60}")
            result = self.download_dataset(dataset_name)
            if result:
                print(f"✅ {dataset_name} 下载完成")
            else:
                print(f"❌ {dataset_name} 下载失败")
        
        print(f"\n{'='*60}")
        print("🎉 所有数据集下载完成!")
        self.list_available_datasets()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集管理工具")
    parser.add_argument("--list", action="store_true", help="列出可用数据集")
    parser.add_argument("--download", type=str, help="下载指定数据集")
    parser.add_argument("--download-all", action="store_true", help="下载所有数据集")
    parser.add_argument("--info", type=str, help="显示数据集信息")
    
    args = parser.parse_args()
    
    manager = DatasetManager()
    
    if args.list:
        manager.list_available_datasets()
    
    elif args.download:
        result = manager.download_dataset(args.download)
        if result:
            print(f"\n✅ 数据集已保存到: {result}")
        else:
            print(f"\n❌ 数据集下载失败")
    
    elif args.download_all:
        manager.download_all_datasets()
    
    elif args.info:
        info = manager.get_dataset_info(args.info)
        if info:
            print(f"\n📋 数据集信息: {args.info}")
            print("=" * 40)
            for key, value in info.items():
                print(f"{key}: {value}")
        else:
            print(f"❌ 未找到数据集信息: {args.info}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()