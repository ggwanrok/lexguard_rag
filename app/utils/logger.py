import logging
import sys

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("lexguard")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger

logger = setup_logger()
