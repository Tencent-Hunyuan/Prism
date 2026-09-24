import sys
import pdb
import os

import logging
from loguru import logger
import random


def main_print(content):
    if int(os.environ["RANK"]) <= 0:
        print(content)


# ForkedPdb().set_trace()
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def empty_logger():
    logger = logging.getLogger("hymm_empty_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


def logger_filter(name):
    def filter_(record):
        return record["extra"].get("name") == name
    return filter_


def setup_logger(exp_dir):
    if int(os.environ["RANK"]) <= 0:
        logger.add(os.path.join(exp_dir, "train.log"), level="DEBUG", colorize=False, backtrace=True,
                   diagnose=True, encoding="utf-8", filter=logger_filter("train"))
        logger.add(os.path.join(exp_dir, "val.log"), level="DEBUG", colorize=False, backtrace=True,
                   diagnose=True, encoding="utf-8", filter=logger_filter("val"))
        train_logger = logger.bind(name="train")
        val_logger = logger.bind(name="val")
    else:
        val_logger = train_logger = empty_logger()

    train_logger.info(f"Experiment directory created at: {exp_dir}")

    return train_logger, val_logger


