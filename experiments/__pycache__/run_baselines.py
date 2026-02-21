# experiments/run_baselines.py

from src.data_loader import load_credit_data, get_feature_target_split
from src.split import train_test_split_data
from src.preprocessing import scale_features
from src.baselines import train_logistic_regression, train_decision_tree
from src.evaluate import evaluate_classification


def main():
    df = load_credit_data()
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Logistic Regression (scaled)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    lr_metrics = evaluate_classification(lr_model, X_test_scaled, y_test)

    # Decision Tree (unscaled)
    dt_model = train_decision_tree(X_train, y_train)
    dt_metrics = evaluate_classification(dt_model, X_test, y_test)

    print("\n=== BASELINE MODEL RESULTS ===")
    print("\nLogistic Regression:")
    for k, v in lr_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nDecision Tree:")
    for k, v in dt_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
