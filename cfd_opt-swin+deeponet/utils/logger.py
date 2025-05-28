"""
Logger Utility for DeepONet-CFD Project

This module provides a function to set up a logger for logging training
information to both the console and a log file.

Functions:
- setup_logger: Configures a logger with console and file handlers.
"""
import logging

def setup_logger(save_folder):
    
    logger = logging.getLogger("CFD_Optimizer")
    logger.setLevel(logging.INFO)

   
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

   
    file_handler = logging.FileHandler(f"{save_folder}/training.log")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
