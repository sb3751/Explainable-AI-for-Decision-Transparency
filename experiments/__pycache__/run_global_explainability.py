# experiments/run_global_explainability.py

from src.data_loader import load_credit_data, get_feature_target_split
from src.split import train_test_split_data
from src.blackbox import train_gradient_boosting
from src.explain_shap import compute_shap_values, plot_global_summary


def main():
    df = load_credit_data()
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Train best-performing model
    gb_model = train_gradient_boosting(X_train, y_train)

    # Compute SHAP values on test data
    shap_values = compute_shap_values(gb_model, X_test)

    # Global SHAP summary
    output_path = plot_global_summary(
        shap_values,
        X_test,
        model_name="gradient_boosting"
    )

    print("\n=== GLOBAL SHAP COMPLETED ===")
    print(f"Summary plot saved to: {output_path}")


if __name__ == "__main__":
    main()
