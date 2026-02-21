# src/explain_lime.py

import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path

from src.config import FIGURES_DIR


def explain_instance_lime(model, X_train, X_instance, feature_names, class_names, instance_id):
    """
    Generate and save a LIME explanation for a single instance.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    explanation = explainer.explain_instance(
        X_instance.values[0],
        model.predict_proba,
        num_features=10
    )

    fig = explanation.as_pyplot_figure()
    output_path = FIGURES_DIR / f"lime_instance_{instance_id}.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return output_path
