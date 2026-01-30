"""PaddleOCR 模型训练脚本
========================
此脚本用于训练或微调 PaddleOCR 模型以识别特定字符集（例如中文）。

如何使用:
1. 准备您的数据集。数据集应包含一个标签文件和对应的图像文件夹。
   标签文件格式 (例如 a.jpg\t"文字"):
   - `train_data/train_list.txt`
   - `val_data/val_list.txt`
2. 根据需要修改 `config/settings.py` 中的 `PaddleOCRTrainingConfig`。
3. 运行命令:
   python main.py train_paddle --train-label-file "path/to/train_list.txt" --val-label-file "path/to/val_list.txt"
"""
import os
import sys
import subprocess
import logging
import locale
from pathlib import Path
import yaml
from config import settings

def train_paddle_model(
    train_label_file: str,
    val_label_file: str,
    config: settings.ocr.paddle,
    path_config: settings.paths
):
    """
    执行 PaddleOCR 模型的训练。

    参数:
        train_label_file (str): 训练集标签文件的路径。
        val_label_file (str): 验证集标签文件的路径。
        config (PaddleOCRTrainingConfig): PaddleOCR 训练配置。
        path_config (PathConfig): 项目路径配置。
    """
    logger = logging.getLogger(__name__)
    logger.info("--- 开始 PaddleOCR 模型训练 ---")

    train_script_path = os.path.join(path_config.base_dir, "libs", "tools", "train.py")
    if not os.path.exists(train_script_path):
        logger.error(f"找不到训练脚本: {train_script_path}")
        return {"success": False, "error": "train.py not found in libs/tools directory."}

    # --- 路径和配置准备 ---
    abs_train_label_file = Path(train_label_file).resolve().as_posix()
    abs_val_label_file = Path(val_label_file).resolve().as_posix()
    train_data_dir = Path(train_label_file).parent.resolve().as_posix()
    val_data_dir = Path(val_label_file).parent.resolve().as_posix()
    char_dict_path = (Path(path_config.base_dir) / config.char_dict_path).resolve().as_posix()
    # 使用重命名后的 training_output_dir
    save_model_dir = (Path(path_config.base_dir) / config.training_output_dir).resolve().as_posix()
    
    config_dir = Path(path_config.base_dir) / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "paddle_train_temp.yml"

    # --- 从模板加载并动态填充配置 ---
    template_path = config_dir / "paddle_ocr_template.yml"
    if not template_path.exists():
        logger.error(f"找不到 PaddleOCR 配置模板文件: {template_path}")
        return {"success": False, "error": "PaddleOCR config template not found."}

    with open(template_path, 'r', encoding='utf-8') as f:
        template_config = yaml.safe_load(f)

    # 填充通用动态值
    template_config['Global']['use_gpu'] = config.use_gpu
    template_config['Global']['epoch_num'] = config.epoch
    template_config['Global']['save_model_dir'] = save_model_dir
    template_config['Global']['character_dict_path'] = char_dict_path
    template_config['Global']['max_text_length'] = config.max_text_length
    template_config['Global']['use_space_char'] = config.use_space_char

    # --- 正确处理预训练和恢复训练 ---
    # 清空模板中的默认值
    template_config['Global']['pretrained_model'] = None
    template_config['Global']['checkpoints'] = None

    if config.resume_checkpoint_path:
        # 恢复训练: 设置 checkpoints，这将加载模型、优化器和epoch信息
        resume_path = (Path(path_config.base_dir) / config.resume_checkpoint_path).resolve().as_posix()
        template_config['Global']['checkpoints'] = resume_path
        logger.info(f"配置为从检查点恢复训练: {resume_path}")
    elif config.model_dir:
        # 微调: 设置 pretrained_model，只加载模型权重，从epoch 0开始
        pretrained_path = (Path(path_config.base_dir) / config.model_dir).resolve().as_posix()
        template_config['Global']['pretrained_model'] = pretrained_path
        logger.info(f"配置为从预训练模型开始训练: {pretrained_path}")
    else:
        logger.info("配置为从头开始训练。" )

    template_config['Optimizer']['lr']['learning_rate'] = float(config.learning_rate)

    template_config['Train']['dataset']['data_dir'] = train_data_dir
    template_config['Train']['dataset']['label_file_list'] = [abs_train_label_file]
    template_config['Train']['loader']['batch_size_per_card'] = config.batch_size

    template_config['Eval']['dataset']['data_dir'] = val_data_dir
    template_config['Eval']['dataset']['label_file_list'] = [abs_val_label_file]

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(template_config, f, allow_unicode=True, sort_keys=False)
        
    logger.info(f"PaddleOCR 训练配置文件已生成: {config_path}")

    python_executable = sys.executable
    env = os.environ.copy()
    libs_path = str(Path(path_config.base_dir) / 'libs')
    env['PYTHONPATH'] = libs_path + os.pathsep + env.get('PYTHONPATH', '')
    
    command = [
        python_executable, "-m", "paddle.distributed.launch",
        "--gpus", "0",
        str(train_script_path),
        f"-c={config_path}"
    ]
    
    logger.info(f"即将执行训练命令: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, encoding=locale.getpreferredencoding(), 
            errors='replace', env=env
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        if rc != 0:
            logger.error(f"PaddleOCR 训练失败，返回码: {rc}")
            return {"success": False, "error": f"Training process failed with exit code {rc}."}

        logger.info("--- PaddleOCR 模型训练成功 ---")
        
        # --- 自动导出模型 ---
        logger.info("--- 开始自动导出推理模型 ---")
        export_script_path = str(Path(path_config.base_dir) / "libs" / "tools" / "export_model.py")
        if not os.path.exists(export_script_path):
            logger.error(f"找不到导出脚本: {export_script_path}")
            return {"success": True, "message": "训练成功，但导出脚本未找到，跳过导出。"}

        best_model_path = (Path(save_model_dir) / "best_accuracy").as_posix()
        inference_save_dir = (Path(path_config.base_dir) / config.inference_model_dir).resolve().as_posix()

        # For exporting, we use the same temp config but override checkpoints to the best model
        template_config['Global']['checkpoints'] = best_model_path
        template_config['Global']['pretrained_model'] = None # Ensure pretrained_model is not set
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(template_config, f, allow_unicode=True, sort_keys=False)

        export_command = [
            python_executable,
            export_script_path,
            f"-c={config_path}",
            "-o", f"Global.save_inference_dir={inference_save_dir}"
        ]

        logger.info(f"即将执行导出命令: {' '.join(export_command)}")
        export_result = subprocess.run(
            export_command, capture_output=True, text=True,
            encoding=locale.getpreferredencoding(), errors='replace',
            env=env, check=True
        )
        logger.info("--- 推理模型导出成功 ---")
        logger.info(f"导出日志:\n{export_result.stdout}")
        return {"success": True, "message": f"模型训练和导出均成功。推理模型已保存至 {config.inference_model_dir}"}

    except subprocess.CalledProcessError as e:
        logger.error(f"一个子进程失败，返回码: {e.returncode}")
        logger.error(f"错误日志:\n{e.stderr or e.stdout}")
        return {"success": False, "error": f"一个子进程失败: {e.stderr or e.stdout}"}
    except FileNotFoundError:
        logger.error(f"命令执行失败。无法找到 '{python_executable}' 或 '{train_script_path}'。")
        return {"success": False, "error": "Python executable or train script not found."}
    except Exception as e:
        logger.error(f"发生未知错误: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

if __name__ == '__main__':
    pass
