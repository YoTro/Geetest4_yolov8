"""PaddleOCR 模型训练脚本
========================
此脚本用于训练或微调 PaddleOCR 模型以识别特定字符集（例如中文）。

如何使用:
1. 准备您的数据集。数据集应包含一个标签文件和对应的图像文件夹。
   标签文件格式 (例如 a.jpg	"文字"):
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
        logger.error("请确保 'libs/tools/train.py' 文件存在。")
        return {"success": False, "error": "train.py not found in libs/tools directory."}

    # 获取绝对路径并转换为
    abs_train_label_file = os.path.abspath(train_label_file).replace('\\', '/')
    abs_val_label_file = os.path.abspath(val_label_file).replace('\\', '/')
    train_data_dir = os.path.dirname(abs_train_label_file).replace('\\', '/')
    val_data_dir = os.path.dirname(abs_val_label_file).replace('\\', '/')
    char_dict_path = os.path.join(path_config.base_dir, config.char_dict_path).replace('\\', '/')
    save_model_dir = os.path.join(path_config.base_dir, config.trained_model_dir).replace('\\', '/')
    config_dir = os.path.join(path_config.base_dir, "config")
    os.makedirs(config_dir, exist_ok=True) # 确保目录存在
    config_path = os.path.join(config_dir, "paddle_train_temp.yml")

    # --- 从模板加载并动态填充配置 ---
    template_path = os.path.join(config_dir, "paddle_ocr_template.yml")
    if not os.path.exists(template_path):
        logger.error(f"找不到 PaddleOCR 配置模板文件: {template_path}")
        return {"success": False, "error": "PaddleOCR config template not found."}

    with open(template_path, 'r', encoding='utf-8') as f:
        template_config = yaml.safe_load(f)

    # 填充动态值
    template_config['Global']['use_gpu'] = config.use_gpu
    template_config['Global']['epoch_num'] = config.epoch
    template_config['Global']['save_model_dir'] = save_model_dir
    template_config['Global']['character_dict_path'] = char_dict_path
    if config.model_dir:
        template_config['Global']['checkpoints'] = os.path.join(path_config.base_dir, config.model_dir).replace('\\', '/') # Use model_dir as checkpoints for pretrained model

    template_config['Optimizer']['lr']['learning_rate'] = float(config.learning_rate)

    template_config['Train']['dataset']['data_dir'] = train_data_dir
    template_config['Train']['dataset']['label_file_list'] = [abs_train_label_file]
    template_config['Train']['loader']['batch_size_per_card'] = config.batch_size

    template_config['Eval']['dataset']['data_dir'] = val_data_dir
    template_config['Eval']['dataset']['label_file_list'] = [abs_val_label_file]

    # 保存临时配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(template_config, f, allow_unicode=True, sort_keys=False)
        
    logger.info(f"PaddleOCR 训练配置文件已生成: {config_path}")

    # 使用 sys.executable 来确保使用当前环境的 Python 解释器
    python_executable = sys.executable

    # 构建训练命令
    # 为了让 tools/train.py 找到 ppocr 模块，需要将 libs 目录添加到 PYTHONPATH
    env = os.environ.copy()
    libs_path = os.path.join(path_config.base_dir, 'libs')
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = libs_path + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = libs_path
    
    # 重新构建命令，使用 paddle.distributed.launch
    command = [
        python_executable,
        "-m", "paddle.distributed.launch",
        "--gpus", "0", # 默认使用第一块GPU
        train_script_path,
        f"-c={config_path}",
        f"-o",
        f"Global.pretrained_model={config.model_dir if config.model_dir else ''}" # Ensure model_dir is used correctly
    ]
    # Update pretrained_model path if config.model_dir is set
    if config.model_dir:
        pretrained_path = Path(path_config.base_dir) / config.model_dir
        command[-1] = f"Global.pretrained_model={pretrained_path.as_posix()}"


    logger.info(f"即将执行训练命令: {' '.join(command)}")
    
    try:
        # 使用 subprocess.Popen 来执行命令，并传入修改后的环境变量
        # 使用 locale.getpreferredencoding() 来获取系统默认编码，以避免在Windows上出现解码错误
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding=locale.getpreferredencoding(), 
            errors='replace', # 如果有无法解码的字符，用'?'替换
            env=env
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
        
        # --- 新增：自动导出模型 ---
        logger.info("--- 开始自动导出推理模型 ---")
        
        export_script_path = os.path.join(path_config.base_dir, "libs", "tools", "export_model.py")
        if not os.path.exists(export_script_path):
            logger.error(f"找不到导出脚本: {export_script_path}")
            return {"success": True, "message": "训练成功，但导出脚本未找到，跳过导出。"}

        # 使用训练好的最佳模型作为导出源
        # 注意: PaddleOCR 保存的最佳模型通常在 'best_accuracy'，而不是 'student'
        best_model_path = os.path.join(save_model_dir, "best_accuracy").replace('\\', '/')
        inference_save_dir = os.path.join(path_config.base_dir, config.inference_model_dir).replace('\\', '/')

        export_command = [
            python_executable,
            export_script_path,
            f"-c={config_path}",
            f"-o",
            f"Global.checkpoints={best_model_path}",
            f"Global.save_inference_dir={inference_save_dir}"
        ]

        logger.info(f"即将执行导出命令: {' '.join(export_command)}")
        try:
            # 使用 subprocess.run 因为导出过程通常较快且我们关心其完整输出
            export_result = subprocess.run(
                export_command,
                capture_output=True,
                text=True,
                encoding=locale.getpreferredencoding(),
                errors='replace',
                env=env,
                check=True # 如果返回非零退出码，则会引发 CalledProcessError
            )
            logger.info("--- 推理模型导出成功 ---")
            logger.info(f"导出日志:\n{export_result.stdout}")
            return {"success": True, "message": f"模型训练和导出均成功。推理模型已保存至 {config.inference_model_dir}"}
        
        except subprocess.CalledProcessError as e:
            logger.error(f"推理模型导出失败，返回码: {e.returncode}")
            logger.error(f"导出错误日志:\n{e.stderr}")
            return {"success": False, "error": f"模型训练成功，但导出失败: {e.stderr}"}
        except Exception as e:
            logger.error(f"导出过程中发生未知错误: {e}", exc_info=True)
            return {"success": False, "error": f"模型训练成功，但导出时发生未知错误: {str(e)}"}

    except FileNotFoundError:
        logger.error(f"命令执行失败。无法找到 '{python_executable}' 或 '{train_script_path}'。")
        return {"success": False, "error": "Python executable or train script not found."}
    except Exception as e:
        logger.error(f"PaddleOCR 模型训练期间发生未知错误: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

if __name__ == '__main__':
    pass