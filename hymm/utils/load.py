from hymm.models.modules import MOVABridge


def load_ti2va_wan_transformer(
        args,
        factor_kwargs=None,
        dit_model_name_or_path=None,
        logger=None,
):
    torch_dtype = factor_kwargs["dtype"]
    bridge, extra_components = MOVABridge.from_mova_pretrained(args.training_config.pretrained_model_name_or_path, torch_dtype=torch_dtype, logger=logger)
    return bridge, extra_components


def get_no_split_modules(transformer):
    if isinstance(transformer, MOVABridge):
        return MOVABridge.get_fsdp_no_split_modules()
    else:
        raise ValueError(f"Unsupported transformer type: {type(transformer)}")
