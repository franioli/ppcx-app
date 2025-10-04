import logging

from .functions.logger import setup_logger

logger = setup_logger(level=logging.INFO, name="ppcx")
