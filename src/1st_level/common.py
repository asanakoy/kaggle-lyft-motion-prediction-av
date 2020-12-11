import yaml


def normalize_experiment_name(experiment_name: str):
    if experiment_name.startswith("experiments/"):
        experiment_name = experiment_name[len("experiments/") :]
    if experiment_name.endswith(".yaml"):
        experiment_name = experiment_name[: -len(".yaml")]

    return experiment_name


def load_config_data(experiment_name: str) -> dict:
    with open(f"experiments/{experiment_name}.yaml") as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


MODEL_TYPE_SEGMENTATION = "segmentation"
MODEL_TYPE_REGRESSION = "regression"
MODEL_TYPE_SEGMENTATION_AND_REGRESSION = "segmentation_and_regression"
MODEL_TYPE_REGRESSION_MULTI_MODE = "regression_multi_mode"
MODEL_TYPE_REGRESSION_MULTI_MODE_AUX_OUT = "regression_multi_mode_aux_out"
MODEL_TYPE_REGRESSION_MULTI_MODE_WITH_MASKS = "regression_multi_mode_with_masks"
MODEL_TYPE_REGRESSION_MULTI_MODE_I4X = "regression_multi_mode_i4x"
MODEL_TYPE_ATTENTION = "attention"
MODEL_TYPE_REGRESSION_MULTI_MODE_WITH_OTHER_AGENTS_INPUTS = "reg_other_inputs"
MODEL_TYPE_REGRESSION_MULTI_MODE_EMB = "regression_multi_mode_emb"
