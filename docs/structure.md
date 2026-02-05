# 项目结构

本项目经过了全面的重构，采用现代化的、简洁的模块化设计，以提高代码的可读性、可维护性和可扩展性。

## 顶层目录结构

```
Geetest4_yolov8/
│
├── config/
│   ├── paddle_ocr_template.yml # PaddleOCR训练配置模板
│   └── settings.py             # 唯一的项目配置文件
│
├── core/
│   ├── captcha_processor.py    # 核心处理器，整合所有逻辑
│   ├── gt4.py                  # 极验v4底层API交互
│   ├── manual_fallback.py      # 人工验证GUI函数
│   ├── matcher.py              # 高级图像匹配器 (GrabCut, minAreaRect, Hungarian)
│   ├── paddle_recognizer.py    # PaddleOCR文字识别封装
│   ├── trocr_recognizer.py     # TrOCR文字识别封装
│   └── yolo_inference.py       # YOLOv8推理封装
│
├── data/                       # 数据根目录
│   ├── dataset/                # 处理后数据集 (YOLO, PaddleOCR, TrOCR)
│   ├── models/                 # 训练好的模型文件
│   └── raw/                    # 原始数据
│
├── docs/                       # 项目文档
│
├── libs/                       # 第三方库和子模块 (如 ppocr)
│
├── logs/                       # 日志与调试图片输出
│
├── training/
│   ├── data_collector.py       # 数据收集
│   ├── dataset_preparation.py  # 数据集准备与增强
│   ├── run_matching.py         # [测试] ImageMatcher匹配测试脚本
│   ├── semi_auto_labeler.py    # 半自动标注工具
│   ├── synthetic_data_generator.py # 合成数据生成器
│   ├── text_extractor.py       # 文字区域提取
│   ├── train_paddleocr.py      # PaddleOCR模型训练
│   ├── train_trocr.py          # TrOCR模型训练
│   └── train_yolo.py           # YOLO模型训练
│
├── main.py                     # 主程序入口
├── README.md
└── requirements.txt            # Python依赖包列表
```

## 各模块核心职责

### `main.py`
项目的**统一命令行接口 (CLI)**。它使用 `argparse` 的子命令系统来解析用户输入，并将任务分发到相应的模块进行处理。它是所有操作的起点。

### `config/`
项目的配置中心。
- `settings.py`: **唯一的配置源**。使用 `dataclasses` 定义了所有可配置的参数（如路径、API密钥、模型超参数等），并创建一个全局可用的 `settings` 对象。

### `core/`
负责处理验证码的核心业务逻辑。
- `captcha_processor.py`: 项目的**大脑**。它是一个类，封装了处理验证码的完整状态和流程，实现了文本匹配与全局高级图像匹配的混合策略。
- `gt4.py`: 一个独立的类，封装了与极验v4后端进行底层网络通信和复杂加密/解密的所有细节。
- `manual_fallback.py`: 无状态的函数式模块，提供启动手动验证 GUI 的单一功能。
- `matcher.py`: 实现 `ImageMatcher` 类，用于图像特征提取和相似度匹配。包含针对复杂背景字符的 GrabCut 分割、minAreaRect 校正、多角度旋转及匈牙利算法匹配等高级功能。
- `paddle_recognizer.py`: PaddleOCR 文字识别的封装类，提供 PaddleOCR 模型加载和识别功能。
- `trocr_recognizer.py`: TrOCR 文字识别的封装类，提供 TrOCR 模型加载和识别功能。
- `yolo_inference.py`: 无状态的函数式模块，提供 YOLO 模型推理的单一功能。

### `data/`
项目的数据根目录，按照用途清晰地组织数据：
- `raw/`: 存放所有原始、未处理的数据。
- `dataset/`: 存放所有经过处理、可直接用于模型训练的数据集。
- `models/`: 存放预训练模型、微调后的模型权重文件。

### `libs/`
存放第三方库的源代码或本项目依赖的子模块。
- `ppocr/`: PaddleOCR 的核心代码库。

### `logs/`
负责配置日志系统，并存放日志文件和调试图片。

### `training/`
包含与模型训练生命周期相关的所有功能，所有模块都已重构为函数式。
- `data_collector.py`: 提供用于收集新训练样本的函数。
- `dataset_preparation.py`: 提供从原始数据创建完整YOLO数据集的函数。
- `run_matching.py`: 用于独立测试 `ImageMatcher` 功能的脚本。
- `semi_auto_labeler.py`: 半自动标注工具。
- `synthetic_data_generator.py`: 生成合成验证码图像。
- `text_extractor.py`: 从原始图片中提取文字区域的智能脚本。
- `train_paddleocr.py`, `train_trocr.py`, `train_yolo.py`: 分别封装了 PaddleOCR, TrOCR, YOLO 模型的训练函数。

### `utils/`
包含一系列无状态的、可被项目全局复用的纯函数模块（如图像处理、坐标转换等）。
