# src/preprocessing.py

from sklearn.preprocessing import StandardScaler


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    Used ONLY for models that require scaling (e.g., Logistic Regression).
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
