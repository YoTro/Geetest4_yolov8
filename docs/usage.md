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
  ```
- **主要选项**:
  - `--mode [auto|manual]`: 指定运行模式 (默认: `auto`)。
  - `--yolo-model <path>`: **(可选)** 用于覆盖配置中定义的 YOLO 模型路径。
  - `--ocr-engine [trocr|paddle]`: **(可选)** 用于覆盖配置中定义的 OCR 引擎类型。
  - `--trocr-model <name_or_path>`: **(可选)** 当 OCR 引擎为 `trocr` 时，指定 TrOCR 模型。
  - `--paddle-model-dir <path>`: **(可选)** 当 OCR 引擎为 `paddle` 时，指定 PaddleOCR 模型目录。

---

### `prepare`
从原始数据文件夹准备标准的YOLOv8格式数据集。

```bash
python3 main.py prepare --source "path/to/your_raw_data"
```
- **主要选项**:
  - `--source <path>`: **(必需)** 原始数据目录。
  - `--output <path>`: 数据集输出目录 (默认: `data/dataset/yolo`)。
  - `--train-ratio <float>`: 训练集比例 (默认: 0.8)。
  - `--augment <int>`: 数据增强倍数 (默认: 1，不增强)。

---

### `train`
使用准备好的数据集训练一个新的YOLOv8模型。

```bash
python3 main.py train --data "data/dataset/dataset.yaml" --epochs 100
```

---

### `train_trocr`
微调一个TrOCR模型，用于识别文字。

在运行此命令前，您需要准备一个TrOCR数据集。

```bash
python3 main.py train_trocr --dataset-dir "data/dataset/trocr" --epochs 5
```

---

### `train_paddle`
训练PaddleOCR文字识别模型。

```bash
python3 main.py train_paddle \
    --train-label-file "data/dataset/paddle/rec_gt_train.txt" \
    --val-label-file "data/dataset/paddle/rec_gt_val.txt"
```

---

## 开发者与调试命令

### `run_matching` (独立测试脚本)
除了 `main.py` 的命令外，项目还提供了独立的匹配测试脚本 `training/run_matching.py`，用于直接调试 `ImageMatcher` 的功能。

- **运行方式**: 此脚本通过 `python -m` 运行。
  ```bash
  # 运行基础匹配测试 (一对一匹配)
  python3 -m training.run_matching simple

  # 运行高级匹配测试 (多对多, GrabCut + 匈牙利算法)
  python3 -m training.run_matching advanced
  ```
- **注意**: 在运行高级测试前，请确保已在 `training/run_matching.py` 中配置了正确的测试图片路径。
