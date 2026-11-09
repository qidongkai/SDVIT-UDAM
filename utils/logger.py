"""
Logging utility.

Provides a single logger factory that is shared across training,
evaluation and deployment scripts. The logger writes to stdout with a
uniform timestamped format and optionally mirrors the output to a file
when a log path is supplied.
"""
import logging
import sys


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name="sdvit", log_path=None, level=logging.INFO):
    """
    Build or retrieve a logger.

    Args:
        name:     logger name; one logger per component is recommended.
        log_path: optional path to a log file. When given, the logger
                  mirrors its output to this file.
        level:    logging level.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when get_logger is called multiple times.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger