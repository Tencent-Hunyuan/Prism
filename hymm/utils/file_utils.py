import logging

CODE_SUFFIXES = {
    ".py",  # Python codes
    ".sh",  # Shell scripts
    ".yaml",
    ".yml",  # Configuration files
}


def empty_logger():
    logger = logging.getLogger("hymm_empty_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger

def rank0_logger(rank):
    if rank == 0:
        from loguru import logger
    else:
        logger = empty_logger()
    return logger
