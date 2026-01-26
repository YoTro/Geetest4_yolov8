# 使用说明

本项目通过 `main.py` 提供了一个基于子命令的、功能强大的命令行接口 (CLI)。

## 基本用法

所有操作都通过以下基本命令结构执行：

```bash
python3 main.py <COMMAND> [OPTIONS]
```

- `<COMMAND>`: **必需参数**，用于指定要执行的任务（如 `run`, `prepare`, `train` 等）。
- `[OPTIONS]`: 每个命令后可跟的附加参数。

您可以使用 `--help` 查看全局选项或某个特定命令的详细帮助，例如：
```bash
python3 main.py --help
python3 main.py run --help
python3 main.py train --help
```

## 主要命令详解

### `run`
运行验证码处理器，这是项目的核心功能。

- **自动模式** (默认使用 `config/settings.py` 中的配置):
  ```bash
  python3 main.py run --mode auto
  ```
- **手动模式** (会弹出GUI窗口):
  ```bash
  python3 main.py run --mode manual
  ```
- **使用自定义模型运行自动模式**:
  你可以通过命令行参数临时覆盖 `config/settings.py` 中定义的模型路径。这在测试不同模型版本或快速切换模型时非常有用。

  ```bash
  # 使用特定的YOLO模型和TrOCR引擎及模型
  python3 main.py run --mode auto \
      --yolo-model "data/models/my_custom_yolo.pt" \
      --ocr-engine trocr \
      --trocr-model "runs/trocr_train/my_trained_trocr"

  # 使用特定的YOLO模型和PaddleOCR引擎及模型
  python3 main.py run --mode auto \
      --yolo-model "data/models/my_custom_yolo.pt" \
      --ocr-engine paddle \
      --paddle-model-dir "runs/paddle_train/my_trained_paddle_inference"
  ```
- **主要选项**:
  - `--mode [auto|manual]`: 指定运行模式 (默认: `auto`)。
  - `--captcha-id <ID>`: 临时覆盖配置文件中的 `captcha_id`。
  - `--yolo-model <path>`: **(可选)** 用于覆盖 `config/settings.py` 中 `settings.yolo_inference.model_path` 的 YOLO 模型路径。
  - `--ocr-engine [trocr|paddle]`: **(可选)** 用于覆盖 `config/settings.py` 中 `settings.ocr.engine` 的 OCR 引擎类型。
  - `--trocr-model <name_or_path>`: **(可选)** 当 `--ocr-engine` 为 `trocr` 时，用于覆盖 `config/settings.py` 中 `settings.ocr.trocr.model_name` 的 TrOCR 模型名称或本地路径。
  - `--paddle-model-dir <path>`: **(可选)** 当 `--ocr-engine` 为 `paddle` 时，用于覆盖 `config/settings.py` 中 `settings.ocr.paddle.inference_model_dir` 的 PaddleOCR 推理模型目录。

---

### `prepare`
从原始数据文件夹准备标准的YOLOv8格式数据集。

```bash
python3 main.py prepare --source "path/to/your_raw_data"
```
- **目录结构要求**: `source` 目录应包含 `images` 和 `labels` 两个子目录，其中分别存放图片和对应的 `.txt` 标签文件。
- **主要选项**:
  - `--source <path>`: **(必需)** 原始数据目录。
  - `--output <path>`: 数据集输出目录 (默认: `data/dataset/yolo`)。
  - `--train-ratio <float>`: 训练集比例 (默认: 0.8)。
  - `--augment <int>`: 数据增强倍数，`3` 表示将训练集增强至原先的3倍 (默认: 1，不增强)。

---

### `collect`
收集新的验证码图片用于后续的标注和训练。

- **无代理，降频模式** (推荐用于少量收集):
  ```bash
  # 使用默认3秒延迟，收集20个样本
  python3 main.py collect --samples 20
  ```
- **使用代理池（并发模式）** (推荐用于大量收集):
  ```bash
  # 从URL加载代理，并发数为10，收集500个样本
  python3 main.py collect --proxy-source http://myproxy.com/api --samples 500 --workers 10
  ```
- **主要选项**:
  - `--samples <int>`: 目标收集数量。
  - `--proxy-source <URL/path>`: **(可选)** 代理源，可以是返回代理列表的URL或本地文件路径。提供此项将启用代理模式。
  - `--delay <float>`: 无代理模式下，两次请求间的延迟秒数 (默认: 3.0)。
  - `--workers <int>`: 代理模式下的并发线程数 (默认: 10)。

---

### `train`
使用准备好的数据集训练一个新的YOLOv8模型。

```bash
python3 main.py train --data "data/dataset/dataset.yaml" --epochs 100
```
- **主要选项**:
  - `--data <path>`: 数据集的 `dataset.yaml` 配置文件路径。
  - `--model <name>`: 用作训练基础的模型，可以是 `yolov8n.pt` 或之前训练好的模型路径 (默认: `yolov8n.pt`)。
  - `--epochs <int>`: 训练轮数。
  - `--batch <int>`: 批次大小。
  - `--device <id>`: 训练设备 (如 `cpu`, `0`, `0,1`)。

---

### `evaluate`
评估模型在验证集上的性能。

```bash
python3 main.py evaluate --model "best.pt" --data "data/dataset/dataset.yaml"
```
- **主要选项**:
  - `--model <name>`: 要评估的模型文件名 (位于 `data/models/` 目录下)。
  - `--data <path>`: 数据集的 `dataset.yaml` 配置文件路径。

---

### `export`
将训练好的 `.pt` 模型导出为其他格式（如 ONNX）。

```bash
python3 main.py export --model "best.pt" --format onnx
```
- **主要选项**:
  - `--model <name>`: 要导出的模型文件名。
  - `--format <format>`: 目标格式 (如 `onnx`, `tensorrt`)。

---

## 半自动标注与TrOCR工作流

除了核心的点选功能，项目还提供了一套用于**文字识别**的半自动流程。该流程首先使用YOLO模型定位文字，然后使用TrOCR模型识别文字内容。

### 工作流概览

1.  **训练YOLO模型**: 训练一个YOLO模型来**识别图片中的文字区域** (Text Bounding Box)。
2.  **提取文字区域**: 使用 `extract_text_regions` 命令，让训练好的YOLO模型自动从大量原始图片中裁剪出文字区域，生成TrOCR的训练数据。
3.  **标注TrOCR数据**: 手动为上一步裁剪出的图片进行文字标注。
4.  **训练TrOCR模型**: 使用 `train_trocr` 命令，训练一个专门用于识别这些文字的TrOCR模型。
5.  **测试识别效果**: 使用 `recognize` 命令，验证TrOCR模型的识别能力。

### `extract_text_regions`
使用YOLO模型从图像中自动提取文字区域，为TrOCR训练准备数据集。

```bash
python3 main.py extract_text_regions \
    --yolo-model "runs/detect/train/weights/best.pt" \
    --input-dir "data/raw/images" \
    --output-dir "data/dataset_trocr"
```
- **主要选项**:
  - `--yolo-model <path>`: **(必需)** 用于检测文字区域的、训练好的YOLO模型路径。
  - `--input-dir <path>`: 包含原始图像的输入目录。
  - `--output-dir <path>`: 保存裁剪后文字图片的输出目录。

### `train_trocr`
微调一个TrOCR模型，用于识别文字。

在运行此命令前，您需要准备一个TrOCR数据集。目录结构应如下：
```
<dataset_dir>/
├── images/
│   ├── train/
│   │   ├── image1.png
│   │   └── image2.png
│   └── val/
│       └── image3.png
└── labels/
    ├── train.txt
    └── val.txt
```
其中 `labels.txt` 文件的每一行格式为 `图片文件名\t文字内容` (例如: `image1.png\t你好`)。

```bash
python3 main.py train_trocr --dataset-dir "data/dataset/trocr" --epochs 5
```
- **主要选项**:
  - `--dataset-dir <path>`: TrOCR数据集的根目录。
  - `--epochs <int>`: 训练轮数。
  - `--batch-size <int>`: 批次大小。
  - `--learning-rate <float>`: 学习率。
  - `--model-name <name_or_path>`: 基础TrOCR模型名称或路径 (默认: `config/settings.py` 中定义)。
  - `--output-dir <path>`: 模型输出目录 (默认: `config/settings.py` 中定义)。
  - `--device <device>`: 训练设备 (如 `cuda`, `cpu`)。
  - `--resume`: 是否从上一次的断点恢复训练。

---

### `train_paddle`
训练PaddleOCR文字识别模型。

**注意**：此命令需要 PaddleOCR 的训练脚本，并依赖于 `config/paddle_ocr_template.yml`。训练数据准备请参照 PaddleOCR 官方文档或项目中 `training/train_paddleocr.py` 的具体要求。

```bash
python3 main.py train_paddle \
    --train-label-file "data/dataset/paddle/rec_gt_train.txt" \
    --val-label-file "data/dataset/paddle/rec_gt_val.txt"
```
- **主要选项**:
  - `--train-label-file <path>`: **(必需)** 训练集标签文件的路径。
  - `--val-label-file <path>`: **(必需)** 验证集标签文件的路径。

### `recognize`

使用 OCR 模型识别单张图片中的文字。支持 TrOCR 和 PaddleOCR 引擎。



- **使用 TrOCR 引擎识别**:

  ```bash

  # 使用默认的TrOCR模型

  python3 main.py recognize --image "data/raw/images/example.png" --engine trocr

  # 使用自定义的TrOCR模型

  python3 main.py recognize --image "data/raw/images/example.png" --engine trocr --model "runs/trocr_train/my_trained_trocr_model"

  ```

- **使用 PaddleOCR 引擎识别**:

  ```bash

  # 使用PaddleOCR模型 (需要配置config/settings.py中的inference_model_dir)

  python3 main.py recognize --image "data/raw/images/example.png" --engine paddle

  # 使用自定义的PaddleOCR模型

  python3 main.py recognize --image "data/raw/images/example.png" --engine paddle --model "runs/paddle_train/my_trained_paddle_inference"

  ```

- **主要选项**:

  - `--image <path>`: **(必需)** 要识别的图片路径。

  - `--engine [trocr|paddle]`: 要使用的 OCR 引擎 (默认: `trocr`)。

  - `--model <name_or_path>`: 当 `--engine` 为 `trocr` 时，指定 TrOCR 模型名称或本地路径；当 `--engine` 为 `paddle` 时，指定 PaddleOCR 推理模型目录。