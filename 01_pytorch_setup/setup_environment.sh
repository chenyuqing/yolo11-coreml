#!/bin/bash
# 第一步：环境搭建脚本
set -e

echo "🚀 设置 YOLOv11 PyTorch 环境"
echo "=================================="

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📋 检测到 Python 版本: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 版本过低，需要 3.8 或更高版本"
    exit 1
fi

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
else
    echo "✅ 虚拟环境已存在"
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 升级 pip
echo "⬆️  升级 pip..."
pip install --upgrade pip

# 安装依赖
echo "📦 安装依赖包..."
pip install -r requirements.txt

echo ""
echo "✅ 环境搭建完成！"
echo ""
echo "🔍 运行测试:"
echo "   source venv/bin/activate"
echo "   python test_pytorch_model.py"