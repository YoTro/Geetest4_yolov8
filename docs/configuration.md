# 配置详解

项目的所有配置都统一在 `config/settings.py` 文件中进行管理。该文件使用 Python 的 `dataclasses` 来定义结构化的配置对象，使得配置清晰、易于访问且支持类型提示。

项目启动时会创建一个全局的 `settings` 实例，可以在项目的任何地方通过 `from config import settings` 导入和使用。

## 主要配置模块说明

### `PathConfig`
管理项目中的所有文件和目录路径。所有路径都基于项目根目录。

-   `base_dir`: 项目的根目录。
-   `model_dir`: 存放训练好的模型文件的目录 (`data/models/`)。
-   `raw_data_dir`: 存放原始、未处理数据的目录 (`data/raw/`)。
-   `yolo_dataset_dir`, `paddle_dataset_dir`, `trocr_dataset_dir`: 分别存放对应格式数据集的目录。
-   `synthetic_main_data_dir`: 存放合成数据（图像和标签）的目录。
-   `debug_output_dir`: 存放调试图片（如带有点选标记的验证码）的目录 (`logs/debug_output/`)。
-   `dict_image_dir`: `ImageMatcher` 用于生成和缓存特征的字典图片目录。

### `GeetestConfig`
包含所有与极验 API 直接交互所需的参数，如 `captcha_id`, `rsa_public_key` 等。

### `ModeSwitchConfig`
定义了自动模式和手动模式之间切换的阈值。
- `max_auto_failures`: 自动模式连续失败多少次后，自动降级到手动模式。
- `min_success_for_switch`: 手动模式连续成功多少次后，尝试切换回自动模式。

### `LoggingConfig`
配置日志系统的行为，如日志级别、输出位置（控制台/文件）、日志文件大小等。

### `DatasetConfig`
数据集的核心配置。
- `names`: **(关键)** 一个列表，包含了您YOLO模型需要识别的所有类别名称。**此列表必须与您训练模型时使用的类别完全一致。**

### `YOLOInferenceConfig`
定义了 YOLOv8 模型推理（检测）时的参数。
- `model_path`: 默认使用的模型文件名，相对于 `data/models/` 目录。
- `conf`: 置信度阈值。
- `iou`: IoU (交并比) 阈值。

### `OCRConfig` (核心配置)
统一管理所有 OCR 引擎的配置。

-   `engine`: **(核心)** 指定当前使用的 OCR 引擎，可选值有 `'trocr'` 或 `'paddle'`。
-   `trocr`: 一个 `TrOCRTrainingConfig` 实例，包含了 TrOCR 引擎的详细配置。
-   `paddle`: 一个 `PaddleOCRTrainingConfig` 实例，包含了 PaddleOCR 引擎的详细配置。

#### `TrOCRTrainingConfig`
-   `model_name`: 用于微调的基础 TrOCR 模型名称或路径 (Hugging Face ID 或本地路径)。在 `run` 命令中，如果选择了 TrOCR 引擎，此参数也用于指定推理模型。
-   `device`: 训练和推理使用的设备 (`cuda` 或 `cpu`)。
-   `min_auto_confidence`: **(泛化配置)** 用于文本匹配阶段的最低置信度阈值。

#### `PaddleOCRTrainingConfig`
-   `inference_model_dir`: 最终用于推理的 PaddleOCR 模型目录。
-   `use_gpu`: 指定是否使用 GPU (`'1'` 或 `'0'`)。
-   `lang`: 模型支持的语言 (如 `ch`)。
-   `min_auto_confidence`: **(泛化配置)** 用于文本匹配阶段的最低置信度阈值。
-   `similarity_threshold`: 用于特征向量相似度匹配的阈值。

### `ImageMatcherConfig` (核心配置)
定义了高级图像匹配器的相关参数。

- `default_weights`: 一个字典，定义了在计算总相似度时，SSIM、HOG 和投影特征各自的权重。
- `min_match_score`: 最小匹配分数阈值。在 `ques` 图片识别和高级匹配中，低于此分数的匹配被认为是不可靠的。

---
*注意：关于模型训练的详细超参数（如 `YOLOTrainingConfig`），请直接参阅 `config/settings.py` 文件。*
