import logging

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger with the specified name.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)  # Default level
    return logger


def set_global_log_level(level: int):
    """
    Sets the logging level for all loggers in the application.

    Args:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    logging.getLogger().setLevel(level)
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(level)
