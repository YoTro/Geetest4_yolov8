# 配置详解

项目的所有配置都统一在 `config/settings.py` 文件中进行管理。该文件使用 Python 的 `dataclasses` 来定义结构化的配置对象，使得配置清晰、易于访问且支持类型提示。

项目启动时会创建一个全局的 `settings` 实例，可以在项目的任何地方通过 `from config import settings` 导入和使用。

## 配置文件结构 (`config/settings.py`)

配置被组织在多个嵌套的 `dataclass` 中，主入口是 `Settings` 类。

```python
# config/settings.py (结构示例)

# ... (详细结构请参考下方的 "主要配置模块说明" 部分) ...
```

## 主要配置模块说明

### `PathConfig`
管理项目中的所有文件和目录路径。所有路径都基于项目根目录。

-   `base_dir`: 项目的根目录。
-   `data_root_dir`: 所有数据相关的根目录 (`data/`)。
-   `model_dir`: 存放训练好的模型文件（如 YOLO 的 `best.pt`）的目录 (`data/models/`)。
-   `raw_data_dir`: 存放原始、未处理数据（如原始图像、`gt.jsonl`）的目录 (`data/raw/`)。
-   `yolo_dataset_dir`: 存放 YOLOv8 格式数据集的目录 (`data/dataset/yolo/`)。
-   `paddle_dataset_dir`: 存放 PaddleOCR 格式数据集的目录 (`data/dataset/paddle/`)。
-   `trocr_dataset_dir`: 存放 TrOCR 格式数据集的目录 (`data/dataset/trocr/`)。
-   `synthetic_trocr_data_dir`: 存放合成 TrOCR 训练数据（图像和标签）的目录 (`data/raw/synthetic_trocr_data/`)。
-   `background_images_dir`: 存放用于合成数据生成的背景图像的目录 (`data/raw/backgrounds/`)。
-   `log_dir`: 日志文件存放目录 (`logs/`)。
-   `runs_dir`: 存放训练运行结果（如 YOLO 的 `runs/detect`）的目录。
-   `best_model_name`: YOLO 最佳模型的默认文件名 (`best.pt`)。
-   `last_model_name`: YOLO 最新模型的默认文件名 (`last.pt`)。
-   `yolo_dataset_yaml_name`: YOLO 数据集配置文件的默认文件名 (`dataset.yaml`)。
-   `get_model_path(model_name)`: 辅助方法，用于获取指定模型的绝对路径。
-   `get_yolo_dataset_yaml_path()`: 辅助方法，用于获取 YOLO 数据集 `dataset.yaml` 的绝对路径。

**通过 `PathConfig` 属性访问路径的示例:**
```python
# 在其他模块中
from config import settings
import os

yolo_model_path = settings.paths.get_model_path(settings.yolo_inference.model_path)
trocr_dataset_image_dir = os.path.join(settings.paths.base_dir, settings.paths.trocr_dataset_dir, "images")
```

### `GeetestConfig`
包含所有与极验 API 直接交互所需的参数。
- `captcha_id`: 您的目标验证码ID。
- `rsa_public_key`: 用于加密通信的RSA公钥。
- `iv_str_hex`: 用于AES加密的初始化向量 (IV)。

### `ModeSwitchConfig`
定义了自动模式和手动模式之间切换的阈值。
- `max_auto_failures`: 自动模式连续失败多少次后，自动降级到手动模式。
- `min_success_for_switch`: 手动模式连续成功多少次后，尝试切换回自动模式。

### `LoggingConfig`
配置日志系统的行为，如日志级别、输出位置（控制台/文件）、日志文件大小等。

### `DataCollectionConfig`
数据收集相关的配置，如默认输出目录。

### `DatasetConfig`
数据集的核心配置。
- `names`: **(关键)** 一个列表，包含了您YOLO模型需要识别的所有类别名称（例如 `['牛', '马', '羊']`）。**此列表必须与您训练模型时使用的类别完全一致。**

### `YOLOTrainingConfig`
定义了模型训练时的所有超参数，如 `epochs`, `batch`, `imgsz` (图像尺寸), `device` 等。这些参数与 `ultralytics` 库的训练参数对齐。

### `YOLOInferenceConfig`
定义了模型推理（检测）时的参数。
- `model_path`: 默认使用的模型文件名，相对于 `data/models/` 目录 (例如 `best.pt`)。
- `conf`: 置信度阈值。
- `iou`: IoU (交并比) 阈值。

### `TrOCRTrainingConfig`
定义了 TrOCR 模型训练和推理时的相关参数。

-   `encoder_name`: 编码器模型名称 (Hugging Face ID)。
-   `decoder_name`: 解码器模型名称 (Hugging Face ID)。
-   `model_name`: 用于微调的基础 TrOCR 模型名称或路径 (Hugging Face ID 或本地路径)。在 `run` 命令中，如果选择了 TrOCR 引擎，此参数也用于指定推理模型。
-   `output_dir`: TrOCR 模型的训练输出目录。
-   `epochs`: 训练的总轮数。
-   `batch_size`: 训练时的批次大小。
-   `learning_rate`: 训练时的学习率。
-   `device`: 训练和推理使用的设备 (`cuda` 或 `cpu`)。
-   `patience`: 早停机制的耐心值，即在多少个 epoch 内验证集指标没有改善就停止训练。
-   `save_total_limit`: 检查点 (checkpoint) 保存数量的上限。
-   `metric_for_best_model`: 决定哪个指标用于判断最佳模型的验证集指标。

### `PaddleOCRTrainingConfig`
定义了 PaddleOCR 模型训练和推理时的相关参数。

-   `model_dir`: 训练时作为起始的预训练模型目录。
-   `trained_model_dir`: 训练后模型的保存目录，包含检查点 (checkpoints)。
-   `inference_model_dir`: 最终用于推理的 PaddleOCR 模型目录，其中应包含 `inference.json` 和 `inference.pdiparams` 文件。在 `run` 命令中，如果选择了 PaddleOCR 引擎，此参数用于指定推理模型。
-   `use_gpu`: 指定是否使用 GPU (`GPU` 或 `CPU`)。
-   `use_angle_cls`: 是否使用角度分类器。
-   `lang`: 模型支持的语言 (如 `ch`)。
-   `epoch`: 训练的总轮数。
-   `batch_size`: 训练时的批次大小。
-   `learning_rate`: 训练时的学习率。
-   `char_dict_path`: 字符字典文件的路径。

### `OCRConfig`
统一管理所有 OCR 引擎的配置。

-   `engine`: **(核心)** 指定当前使用的 OCR 引擎，可选值有 `trocr` 或 `paddle`。
-   `trocr`: 一个 `TrOCRTrainingConfig` 实例，包含了 TrOCR 引擎的详细配置。
-   `paddle`: 一个 `PaddleOCRTrainingConfig` 实例，包含了 PaddleOCR 引擎的详细配置。

## 关于 `dataset.yaml`
项目遵循 YOLOv8 的标准，使用一个 `dataset.yaml` 文件来告知训练器数据集的位置和类别信息。

- **您无需手动创建此文件。**
- 当您运行 `python3 main.py prepare` 命令时，程序会根据 `config/settings.py` 中 `DatasetConfig` 的 `names` 列表，在您指定的数据集输出目录中**自动生成**这个 `dataset.yaml` 文件。
- `train` 和 `evaluate` 命令会默认使用这个自动生成的文件。