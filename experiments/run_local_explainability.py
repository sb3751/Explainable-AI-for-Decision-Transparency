# experiments/run_local_explainability.py

import numpy as np

from src.data_loader import load_credit_data, get_feature_target_split
from src.split import train_test_split_data
from src.blackbox import train_gradient_boosting
from src.explain_shap import compute_shap_values, plot_local_shap
from src.explain_lime import explain_instance_lime
from src.model_wrappers import PredictProbaWrapper


def main():
    df = load_credit_data()
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Train model
    model = train_gradient_boosting(X_train, y_train)
    wrapped_model = PredictProbaWrapper(model, X.columns.tolist())

    # SHAP values on test set
    shap_values, expected_value = compute_shap_values(model, X_test)


    # Choose representative instance indices
    instance_indices = [
        0,    # typical case
        10,   # likely non-default
        50    # likely default
    ]

    print("\n=== LOCAL EXPLAINABILITY ===")

    for idx in instance_indices:
        print(f"\nExplaining instance {idx}")

        shap_path = plot_local_shap(
            shap_values=shap_values,
            expected_value=expected_value,
            X=X_test,
            instance_index=idx,
            model_name="gradient_boosting"
        )

        lime_path = explain_instance_lime(
            model=wrapped_model,
            X_train=X_train,
            X_instance=X_test.iloc[[idx]],
            feature_names=X.columns.tolist(),
            class_names=["No Default", "Default"],
            instance_id=idx
        )

        print(f"SHAP saved to: {shap_path}")
        print(f"LIME saved to: {lime_path}")


if __name__ == "__main__":
    main()
