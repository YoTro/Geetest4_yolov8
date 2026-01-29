# 安装指南

按照以下步骤来设置和运行本项目。

## 1. 系统要求

- Python 3.8 或更高版本
- `pip` 包管理器 (建议使用 `pip3`)
- 对于 `Linux` 系统，需要 `apt` 包管理器 (或根据您的发行版调整)

## 2. 安装步骤

### Linux / Docker 一键安装 (推荐)
如果您正在使用基于 Debian/Ubuntu 的 Linux 发行版 (或基于此的 Docker 镜像)，您可以运行以下单行命令来安装所有系统和 Python 依赖，并下载预训练的 PaddleOCR 模型。

```bash
apt-get update && apt install vim file unzip python3-tk && \
python3 -m pip install paddlepaddle-gpu==2.6.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ && \
pip3 install -r requirements.txt && \
wget -nc -P data/models https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar && \
tar -xf data/models/ch_PP-OCRv4_rec_train.tar -C data/models/
```
> **注意**:
> - 上述命令中的 `paddlepaddle-gpu` 版本和 CUDA (`cu118`) 版本是示例。请访问 [PaddlePaddle 官网](https://www.paddlepaddle.org.cn/install/quick?doc_type=gpu) 以获取与您环境完全匹配的安装命令。
> - 如果您不使用 GPU，请将 `paddlepaddle-gpu` 替换为 `paddlepaddle`。
> - 运行此命令后，您可以直接跳到 **第五步：检查配置**。

---

### 常规安装步骤

#### 第一步：克隆项目

将项目代码克隆到您的本地计算机：

```bash
git clone <your-repository-url>
cd geetest_automation
```

#### 第二步：创建虚拟环境 (强烈推荐)

为了保持项目依赖的隔离，建议您使用虚拟环境：

```bash
# Windows
py -3 -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 第三步：安装依赖

项目的所有依赖项都记录在 `requirements.txt` 文件中。运行以下命令进行安装：

```bash
pip3 install -r requirements.txt
```
> **注意**: 
> 1.  **Linux 用户**: 推荐您参考上面的 "Linux / Docker 一键安装" 部分，该部分包含了系统依赖和 PaddlePaddle 的特定安装命令。
> 2.  如果在安装 `torch` 或 `torchvision` 时遇到网络问题或版本兼容性问题，建议访问 [PyTorch 官网](https://pytorch.org/get-started/locally/)，根据您的操作系统和CUDA版本（如果使用GPU）获取对应的安装命令。
> 3. 如果您计划使用 **PaddleOCR** 引擎并希望利用 GPU 加速，请务必根据 [PaddlePaddle 官方安装指南](https://www.paddlepaddle.org.cn/install/quick?doc_type=gpu) 进行安装。

#### 第四步：准备模型 (关键步骤)

本项目 `auto` 模式的成功与否**完全取决于您提供的模型**。您需要准备至少两个核心模型：
1.  **YOLOv8 目标检测模型**：用于检测验证码中的文字区域。
2.  **OCR 文字识别模型**：用于识别文字区域中的具体字符（TrOCR 或 PaddleOCR）。

> **Linux 用户注意**: 上方的 "一键安装" 命令已为您下载并解压了官方的 PaddleOCR 预训练模型。

---

#### 4.1 准备 YOLOv8 目标检测模型

1.  **获取模型**: 您需要自行**训练或获取一个专门用于识别验证码文字区域的 YOLOv8 模型**。通用的 COCO 预训练模型 (如 `yolov8n.pt`) **无法**直接识别文字区域。
    *   您可以使用 `python3 main.py train` 命令进行训练。
2.  **放置模型**: 将您训练好的 YOLOv8 模型（通常是 `best.pt`）放置在 `data/models/` 目录下。
3.  **配置模型**: 默认情况下，`config/settings.py` 中的 `YOLOInferenceConfig.model_path` 会查找 `data/models/best.pt`。如果您的模型文件名不是 `best.pt`，请修改该属性。

#### 4.2 准备 OCR 文字识别模型 (TrOCR 或 PaddleOCR)

根据您在 `config/settings.py` 中 `OCRConfig.engine` 的选择，准备相应的 OCR 模型。

##### 如果选择 TrOCR (推荐微调已有的 TrOCR 模型)

1.  **获取模型**:
    *   您可以直接使用 Hugging Face Hub 上预训练的 TrOCR 模型 ID (例如 `microsoft/trocr-base-stage1`)。
    *   **推荐**：使用 `python3 main.py train_trocr` 命令，在特定数据集上微调一个 TrOCR 模型，以获得更好的识别效果。
2.  **放置模型**:
    *   如果您使用的是 Hugging Face 模型 ID，则无需本地文件。
    *   如果您训练了自己的 TrOCR 模型，它会保存到 `runs/trocr_train/final_trocr_model` 目录（可在 `config/settings.py` 中 `TrOCRTrainingConfig.output_dir` 配置）。
3.  **配置模型**:
    *   在 `config/settings.py` 中，修改 `OCRConfig.trocr.model_name` 属性，将其设置为 Hugging Face 模型 ID 或您的本地模型路径。

##### 如果选择 PaddleOCR (推荐微调已有的 PaddleOCR 模型)

1.  **获取模型**:
    *   您可以下载 PaddleOCR 官方提供的预训练模型 (上方的 "一键安装" 命令已包含此步骤)。
    *   **推荐**：使用 `python3 main.py train_paddle` 命令，在特定数据集上微调一个 PaddleOCR 模型，以获得更好的识别效果。
2.  **放置模型**:
    *   下载的 PaddleOCR 预训练模型或您训练后的推理模型，应放置在一个包含 `inference.pdmodel` 和 `inference.pdiparams` 文件的目录下。
3.  **配置模型**:
    *   在 `config/settings.py` 中，修改 `OCRConfig.paddle.inference_model_dir` 属性，指向您的 PaddleOCR 推理模型目录。

---

### 第五步：检查配置

打开 `config/settings.py` 文件，根据您的需求调整配置。最重要的配置项包括：

1.  **`GeetestConfig`**: 确认 `captcha_id` 是否为您的目标验证码ID。
2.  **`DatasetConfig`**: 确保 `names` 列表与您训练的 YOLO 模型的类别名称完全一致。
3.  **`OCRConfig`**:
    *   `engine`: 设置为 `trocr` 或 `paddle`，以选择您希望使用的 OCR 引擎。
    *   `trocr` / `paddle`: 根据您选择的 OCR 引擎，检查并配置相应的 `model_name` (TrOCR) 或 `inference_model_dir` (PaddleOCR)。

完成以上步骤后，您的开发环境已经准备就绪！