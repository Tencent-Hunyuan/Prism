import importlib
import json


def load_caption_processor(name, caption_sample_ratio, logger, kwargs=None):
    processor_spec = importlib.import_module(f"hymm.dataset.caps.{name}")
    cls = processor_spec.CaptionAug

    # Parse kwargs
    if kwargs is None:
        kwargs = {}
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs)

    return cls(caption_sample_ratio=caption_sample_ratio, logger=logger, **kwargs)