"""
Logger Utility for DeepONet-CFD Project

This module provides a function to set up a logger for logging training
information to both the console and a log file.

Functions:
- setup_logger: Configures a logger with console and file handlers.
"""
import logging

def setup_logger(save_folder):
    """设置logger配置"""
    logger = logging.getLogger("CFD_Optimizer")
    logger.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # 创建文件处理器
    file_handler = logging.FileHandler(f"{save_folder}/training.log")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # 添加处理器到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger