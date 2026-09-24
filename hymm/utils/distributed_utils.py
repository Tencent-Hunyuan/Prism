"""
Distributed training utilities.
"""

def rank0_print(logger, msg):
    """
    Print message only from rank 0 process in distributed training.
    
    Args:
        logger: Logger object (should have .info method)
        msg: Message to print
    """
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            if dist.get_rank() == 0:
                logger.info(msg)
        else:
            # If distributed training is not initialized, print normally
            logger.info(msg)
    except (ImportError, AttributeError):
        # If torch.distributed is not available, print normally
        logger.info(msg)

