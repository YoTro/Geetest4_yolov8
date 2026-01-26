"""
主配置文件
=============
该文件是项目中所有配置的唯一真实来源。
它使用嵌套的 dataclasses 来提供一个清晰、类型安全且结构化的配置方案。
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml

# --- 核心配置模块 ---

@dataclass
class PathConfig:
    """定义项目中的所有关键路径"""
    base_dir: str = str(Path(__file__).parent.parent)
    data_root_dir: str = "data" # New: Base for all data
    model_dir: str = "data/models"

    # Specific dataset roots within data_root_dir
    raw_data_dir: str = "data/raw"
    yolo_dataset_dir: str = "data/dataset/yolo"
    paddle_dataset_dir: str = "data/dataset/paddle"
    trocr_dataset_dir: str = "data/dataset/trocr"
    synthetic_trocr_data_dir: str = "data/raw/synthetic_trocr_data" # New path for synthetic data output
    background_images_dir: str = "data/raw/backgrounds" # New path for background images
    
    log_dir: str = "logs"
    runs_dir: str = "runs/detect" # For YOLO training outputs
    
    # 默认模型和数据文件的相对名称
    best_model_name: str = "best.pt"
    last_model_name: str = "last.pt"
    yolo_dataset_yaml_name: str = "dataset.yaml" # Specific to YOLO

    def get_model_path(self, model_name: str) -> str:
        """获取模型文件的绝对路径"""
        return os.path.join(self.base_dir, self.model_dir, model_name)

    def get_yolo_dataset_yaml_path(self) -> str: # Renamed
        """获取YOLO数据集YAML文件的绝对路径"""
        return os.path.join(self.base_dir, self.yolo_dataset_dir, self.yolo_dataset_yaml_name)

    def __post_init__(self):
        """确保所有关键目录都存在"""
        # Create top-level directories
        os.makedirs(os.path.join(self.base_dir, self.data_root_dir), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, self.model_dir), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, self.raw_data_dir), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, self.yolo_dataset_dir), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, self.paddle_dataset_dir), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, self.trocr_dataset_dir), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, self.synthetic_trocr_data_dir), exist_ok=True) # Create synthetic data output dir
        os.makedirs(os.path.join(self.base_dir, self.background_images_dir), exist_ok=True) # Create background images dir
        os.makedirs(os.path.join(self.base_dir, self.log_dir), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, self.runs_dir), exist_ok=True)

        # Create subdirectories specific to YOLO dataset structure
        os.makedirs(os.path.join(self.base_dir, self.yolo_dataset_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, self.yolo_dataset_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, self.yolo_dataset_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, self.yolo_dataset_dir, "labels", "val"), exist_ok=True)

        # Create subdirectories for TrOCR dataset (where text_extractor.py will output)
        os.makedirs(os.path.join(self.base_dir, self.trocr_dataset_dir, "images"), exist_ok=True)


@dataclass
class GeetestConfig:
    """极验验证码相关配置"""
    captcha_id: str = "54088bb07d2df3c46b79f80300b0abbe"
    risk_type: str = "word"
    lang: str = "zh-cn"
    client_type: str = "web"
    request_timeout: int = 30
    retry_times: int = 3
    retry_delay: float = 1.0
    # RSA公钥配置
    rsa_public_key: Dict[str, str] = field(default_factory=lambda: {
        "n": "00C1E3934D1614465B33053E7F48EE4EC87B14B95EF88947713D25EECBFF7E74C7977D02DC1D9451F79DD5D1C10C29ACB6A9B4D6FB7D0A0279B6719E1772565F09AF627715919221AEF91899CAE08C0D686D748B20A3603BE2318CA6BC2B59706592A9219D0BF05C9F65023A21D2330807252AE0066D59CEEFA5F2748EA80BAB81",
        "e": "10001"
    })
    # 默认请求配置
    default_config: Dict[str, Any] = field(default_factory=lambda: {
        "pt": 1,
        "payload_protocol": 1
    })
    # AES加密的初始化向量 (IV)
    iv_str_hex: str = "0000000000000000"

@dataclass
class ModeSwitchConfig:
    """模式切换配置"""
    max_auto_failures: int = 3
    min_success_for_switch: int = 5
    manual_timeout: int = 60

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    log_dir: str = "logs"
    max_file_size_mb: int = 10
    backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

@dataclass
class DataCollectionConfig:
    """数据收集配置"""
    output_dir: str = "data/raw/collected"
    max_storage_gb: int = 10

@dataclass
class YOLOTrainingConfig:
    """YOLO模型训练配置 (与Ultralytics的命令行参数对齐)"""
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    workers: int = 8
    device: str = "cpu"
    optimizer: str = "auto"
    patience: int = 100
    save_period: int = -1
    project: str = "runs/detect"
    name: str = "train"
    exist_ok: bool = False
    val: bool = True
    deterministic: bool = True
    single_cls: bool = False
    rect: bool = False
    resume: bool = False
    amp: bool = True
    fraction: float = 1.0
    profile: bool = False
    pretrained: bool = True
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    data: str = "" # 将由dataset_preparation在运行时填充

@dataclass
class YOLOInferenceConfig:
    """YOLO模型推理配置"""
    model_path: str = "best.pt"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    max_det: int = 1000
    device: str = "cpu"
    augment: bool = False
    agnostic_nms: bool = False
    half: bool = False
    classes: Optional[List[int]] = None

@dataclass
class TrOCRTrainingConfig:
    """TrOCR模型训练配置"""
    encoder_name: str = "google/vit-base-patch16-224-in21k"
    decoder_name: str = "bert-base-chinese"
    model_name: str = "ZihCiLin/trocr-traditional-chinese-baseline" # Base model for fine-tuning
    output_dir: str = "runs/trocr_train"
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-5
    device: str = "cuda" # 'cuda' or 'cpu'
    patience: int = 5 # 早停
    save_total_limit: int = 2 # 限制checkpoint储存
    metric_for_best_model: str = "eval_loss" # 监控指标

@dataclass
class PaddleOCRTrainingConfig:
    """PaddleOCR模型训练配置
    预训练模型下载
    wget -nc -P https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar
    cd ./data/models && tar -xf ch_PP-OCRv4_rec_train.tar && cd ..
    """
    model_dir: Optional[str] = "data/models/ch_PP-OCRv4_rec_train/student"  # 预训练模型目录，训练时作为起始模型
    trained_model_dir: str = "runs/paddle_train/best_model" # 训练后模型的保存目录 (包含checkpoints)
    inference_model_dir: str = "data/models/inference" # 新增：最终用于推理的模型目录，假设训练脚本会导出到这里
    use_gpu: str = "GPU"
    use_angle_cls: bool = True # 是否使用角度分类器
    lang: str = "ch"  # 默认语言
    # 训练超参数
    epoch: int = 500
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # 字典文件路径
    char_dict_path: str = "libs/ppocr/utils/ppocr_keys_v1.txt"

@dataclass
class OCRConfig:
    """OCR引擎的统一配置"""
    engine: str = "trocr"  # 'trocr' or 'paddle'

    # 具体引擎的配置
    trocr: TrOCRTrainingConfig = field(default_factory=TrOCRTrainingConfig)
    paddle: PaddleOCRTrainingConfig = field(default_factory=PaddleOCRTrainingConfig)


@dataclass
class DatasetConfig:
    """数据集配置"""
    # 默认类别名称，应根据您的实际项目进行修改
    names: List[str] = field(default_factory=lambda: [
        'up', 'down', 'left', 'right' 
    ])

@dataclass
class Settings:
    """统一的主配置类"""
    paths: PathConfig = field(default_factory=PathConfig)
    geetest: GeetestConfig = field(default_factory=GeetestConfig)
    mode_switch: ModeSwitchConfig = field(default_factory=ModeSwitchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    yolo_training: YOLOTrainingConfig = field(default_factory=YOLOTrainingConfig)
    yolo_inference: YOLOInferenceConfig = field(default_factory=YOLOInferenceConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    
    # 根级运行时配置
    debug_mode: bool = False
    save_debug_images: bool = False

    def __post_init__(self):
        """在初始化后执行, 确保目录存在"""
        self.paths.__post_init__()

# --- 全局实例 ---
# 提供一个全局可用的配置实例，方便在项目各处直接导入和使用
settings = Settings()