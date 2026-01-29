"""
日志模块
=========

该模块提供一个统一的函数来配置整个应用程序的日志系统。

可以直接从任何地方导入并使用:
`from logs import setup_logging`
"""
import logging
import logging.handlers
from pathlib import Path
from config.settings import LoggingConfig

def setup_logging(config: LoggingConfig):
    """
    根据提供的配置设置根日志记录器。

    Args:
        config: 包含所有日志参数的 LoggingConfig 对象。
    """
    # 获取根记录器
    root_logger = logging.getLogger()
    
    # 清除任何现有的处理器，以避免重复日志
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 设置日志级别
    log_level = getattr(logging, config.level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # 创建格式化器
    formatter = logging.Formatter(config.log_format)

    # 1. 配置控制台处理器
    if config.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 2. 配置文件处理器
    if config.log_to_file:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # 主日志文件处理器
        main_log_path = log_dir / "main.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_path,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        main_handler.setLevel(log_level)
        main_handler.setFormatter(formatter)
        root_logger.addHandler(main_handler)

        # 错误日志文件处理器 (只记录ERROR及以上级别)
        error_log_path = log_dir / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_path,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)

    # 初始日志消息
    logging.info("日志系统配置完成。")