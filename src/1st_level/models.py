import models_reg


def build_model(model_info, cfg):
    model_name: str = model_info.model
    if model_name.startswith('LyftMultiModel') and hasattr(models_reg, model_name):
        model = getattr(models_reg, model_name)(cfg=cfg)
    else:
        raise RuntimeError("Invalid model name")

    return model
