# experiments/run_data_check.py

from src.data_loader import load_credit_data, get_feature_target_split
from src.split import train_test_split_data

def main():
    df = load_credit_data()
    X, y = get_feature_target_split(df)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print("\n=== SPLIT CHECK ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")
    print(f"y_test distribution:\n{y_test.value_counts()}")

if __name__ == "__main__":
    main()
