# 项目结构

本项目经过了全面的重构，采用现代化的、简洁的模块化设计，以提高代码的可读性、可维护性和可扩展性。

## 顶层目录结构

```
geetest_automation/
│
├── config/
│   ├── __init__.py
│   ├── paddle_ocr_template.yml # PaddleOCR训练配置模板
│   └── settings.py             # 唯一的项目配置文件
│
├── core/
│   ├── __init__.py
│   ├── captcha_processor.py    # 核心处理器，整合所有逻辑
│   ├── gt4.py                  # 极验v4底层API交互
│   ├── manual_fallback.py      # 人工验证GUI函数
│   ├── paddle_recognizer.py    # PaddleOCR文字识别函数
│   ├── trocr_recognizer.py     # TrOCR文字识别函数
│   └── yolo_inference.py       # YOLOv8推理函数
│
├── data/                       # 数据根目录 (详细结构见下方说明)
│   ├── dataset/                # 处理后数据集 (YOLO, PaddleOCR, TrOCR)
│   ├── models/                 # 训练好的模型文件
│   └── raw/                    # 原始数据
│
├── docs/                       # 项目文档
│
├── libs/                       # 第三方库和子模块
│   ├── ppocr/                  # PaddleOCR相关代码
│   └── tools/                  # 其他工具代码
│
├── logs/                       # 日志文件
│
├── training/
│   ├── __init__.py
│   ├── data_collector.py       # 数据收集函数
│   ├── dataset_preparation.py  # 数据集准备与增强函数
│   ├── semi_auto_labeler.py    # 半自动标注工具
│   ├── synthetic_data_generator.py # 合成数据生成器
│   ├── text_extractor.py       # 文字区域提取与TrOCR数据准备
│   ├── train_paddleocr.py      # PaddleOCR模型训练函数
│   ├── train_trocr.py          # TrOCR模型训练函数
│   └── train_yolo.py           # YOLO模型训练与验证函数
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
- `paddle_ocr_template.yml`: PaddleOCR 训练的 YAML 配置文件模板，在训练时会被动态填充。

### `core/`
负责处理验证码的核心业务逻辑。
- `captcha_processor.py`: 项目的**大脑**。它是一个类，封装了处理验证码的完整状态和流程，包括加载验证码、调用 YOLO 模型、OCR 模型、处理手动/自动模式的切换逻辑等。
- `gt4.py`: 一个独立的类，封装了与极验v4后端进行底层网络通信和复杂加密/解密的所有细节。
- `yolo_inference.py`: 无状态的**函数式模块**，提供 YOLO 模型推理的单一功能。
- `manual_fallback.py`: 无状态的**函数式模块**，提供启动手动验证 GUI 的单一功能。
- `trocr_recognizer.py`: TrOCR 文字识别的封装类，提供 TrOCR 模型加载和识别功能。
- `paddle_recognizer.py`: PaddleOCR 文字识别的封装类，提供 PaddleOCR 模型加载和识别功能。

### `data/`
项目的数据根目录，按照用途清晰地组织数据：
- `raw/`: 存放所有原始、未处理的数据。
    - `images/`: 原始验证码图像。
    - `gt.jsonl`: 原始标注文件。
    - `backgrounds/`: 用于合成数据生成的背景图像。
    - `synthetic_trocr_data/`: 合成 TrOCR 训练数据的输出目录。
- `dataset/`: 存放所有经过处理、可直接用于模型训练的数据集。
    - `yolo/`: YOLOv8 格式的数据集（包含 `images/`, `labels/`, `dataset.yaml`）。
    - `paddle/`: PaddleOCR 格式的数据集（例如 `rec_gt_train.txt`, `rec_gt_val.txt`）。
    - `trocr/`: TrOCR 格式的数据集（包含裁剪后的 `images/` 和 `labels.jsonl`）。
- `models/`: 存放预训练模型、微调后的模型权重文件（如 YOLO 的 `best.pt`）。

### `libs/`
存放第三方库的源代码或本项目依赖的子模块，以便更好地管理和集成。
- `ppocr/`: PaddleOCR 的核心代码库，用于其模型训练和推理。
- `tools/`: 其他通用或辅助工具脚本。

### `logs/`
负责配置日志系统。
- `__init__.py`: 包含一个 `setup_logging` 函数，它根据 `config/settings.py` 中的配置，设置Python标准的 `logging` 模块，支持将日志同时输出到控制台和不同的日志文件（如 `main.log`, `error.log`）。

### `training/`
包含与模型训练生命周期相关的所有功能，所有模块都已重构为**函数式**。
- `data_collector.py`: 提供使用代理池或延迟降频策略来安全、高效地收集新训练样本的函数。
- `dataset_preparation.py`: 提供从原始数据（图片+标签）创建完整YOLO数据集的函数，包括数据分割和增强。
- `semi_auto_labeler.py`: 运行半自动标注工具，为裁剪出的文字图片添加标签。
- `synthetic_data_generator.py`: 生成用于 TrOCR 训练的合成验证码图像。
- `text_extractor.py`: 提供一个**智能脚本**，可以从原始图片中提取文字区域，并支持单字符或通用文本区域的标注，为 TrOCR 训练准备数据集。
- `train_paddleocr.py`: 封装了调用 PaddleOCR 官方训练脚本进行 PaddleOCR 模型训练的函数。
- `train_trocr.py`: 封装了调用 `transformers` 库进行 TrOCR 模型微调的函数。
- `train_yolo.py`: 封装了调用 `ultralytics` 库进行 YOLO 模型训练、验证和导出的函数。

### `utils/`
包含一系列无状态的、可被项目全局复用的**纯函数模块**。
- `image_processor.py`: 负责图像处理，如格式转换、下载、裁剪等。
- `coordinate_utils.py`: 负责坐标转换和几何计算。
- `data_augmentation.py`: 基于 `albumentations` 库，提供标准化的数据增强流水线。
- `label_generator.py`: 负责YOLO标签格式的读写，以及 `dataset.yaml` 文件的生成。