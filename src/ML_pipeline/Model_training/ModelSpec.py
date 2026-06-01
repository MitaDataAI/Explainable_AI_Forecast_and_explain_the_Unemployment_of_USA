from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, List

import numpy as np

from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from mlforecast import MLForecast


@dataclass
class ModelSpec:
    name: str
    build_mlf: Callable[[str, Dict[str, Any]], MLForecast]
    pred_col: str
    tunable: bool = False
    param_space: Optional[Dict[str, Iterable[Any]]] = None
    search: str = "random"
    n_iter: int = 50
    tune_cv_windows: int = 6
    tune_every_months: int = 36
    use_conformal_in_tune: bool = False
    fixed_params: Optional[Dict[str, Any]] = None


# =========================================================
# Builders
# =========================================================
def build_ridge_mlf(freq: str, params: Dict[str, Any]) -> MLForecast:
    model = Ridge(**params)
    return MLForecast(
        models={"RIDGE": model},
        freq=freq,
    )


def build_lgbm_mlf(freq: str, params: Dict[str, Any]) -> MLForecast:
    model = LGBMRegressor(**params)
    return MLForecast(
        models={"LGBM": model},
        freq=freq,
    )


# =========================================================
# Param space parser
# =========================================================
def _parse_param_value(value):
    if isinstance(value, dict):
        value_type = value.get("type")

        if value_type == "logspace":
            start_exp = value["start_exp"]
            end_exp = value["end_exp"]
            num = value["num"]
            return list(np.logspace(start_exp, end_exp, num=num))

        raise ValueError(f"Unsupported param space type: {value_type}")

    if isinstance(value, list):
        return value

    raise ValueError(f"Unsupported param specification: {value}")


def _parse_param_space(param_space_cfg: Dict[str, Any]) -> Dict[str, Iterable[Any]]:
    return {
        param_name: _parse_param_value(param_value)
        for param_name, param_value in param_space_cfg.items()
    }


# =========================================================
# Factory
# =========================================================
def build_model_specs(model_config: Dict[str, Any]) -> List[ModelSpec]:
    models_cfg = model_config["models"]
    specs: List[ModelSpec] = []

    builder_map = {
        "RIDGE": build_ridge_mlf,
        "LGBM": build_lgbm_mlf,
    }

    for model_name, cfg in models_cfg.items():
        if not cfg.get("enabled", False):
            continue

        if model_name not in builder_map:
            raise ValueError(f"Unsupported model: {model_name}")

        fixed_params = cfg.get("fixed_params", {}) or {}
        param_space_cfg = cfg.get("param_space", {}) or {}

        spec = ModelSpec(
            name=model_name,
            build_mlf=builder_map[model_name],
            pred_col=cfg["pred_col"],
            tunable=cfg.get("tunable", False),
            param_space=_parse_param_space(param_space_cfg) if param_space_cfg else None,
            search=cfg.get("search", "random"),
            n_iter=cfg.get("n_iter", 50),
            tune_cv_windows=cfg.get("tune_cv_windows", 6),
            tune_every_months=cfg.get("tune_every_months", 36),
            use_conformal_in_tune=cfg.get("use_conformal_in_tune", False),
            fixed_params=fixed_params,
        )

        specs.append(spec)

    return specs