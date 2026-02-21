# src/explain_shap.py

import shap
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from src.config import FIGURES_DIR


def compute_shap_values(model, X):
    """
    Compute SHAP values for tree-based models.
    Returns raw SHAP values and expected value.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value
    return shap_values, expected_value


def plot_global_summary(shap_values, X, model_name):
    """
    Generate and save global SHAP summary plot.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    output_path = FIGURES_DIR / f"shap_summary_{model_name}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    return output_path


def plot_local_shap(
    shap_values,
    expected_value,
    X,
    instance_index,
    model_name
):
    """
    Generate and save a local SHAP waterfall plot
    compatible with NumPy-based SHAP outputs.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build Explanation object from legacy format
    explanation = shap.Explanation(
        values=shap_values[instance_index],
        base_values=expected_value,
        data=X.iloc[instance_index].values,
        feature_names=X.columns.tolist()
    )

    shap.plots.waterfall(
        explanation,
        show=False
    )

    output_path = FIGURES_DIR / f"shap_instance_{model_name}_{instance_index}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    return output_path
