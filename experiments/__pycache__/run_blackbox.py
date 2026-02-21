# experiments/run_blackbox.py

from src.data_loader import load_credit_data, get_feature_target_split
from src.split import train_test_split_data
from src.blackbox import train_random_forest, train_gradient_boosting
from src.evaluate import evaluate_classification


def main():
    df = load_credit_data()
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_classification(rf_model, X_test, y_test)

    # Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_metrics = evaluate_classification(gb_model, X_test, y_test)

    print("\n=== BLACK-BOX MODEL RESULTS ===")

    print("\nRandom Forest:")
    for k, v in rf_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nGradient Boosting:")
    for k, v in gb_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
