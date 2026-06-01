from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    # 1. 데이터셋 로드
    data = load_breast_cancer()
    X = data.data
    y = data.target

    print("Dataset: Breast Cancer Wisconsin")
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print("Class names:", data.target_names)

    # 2. train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain data size:", X_train.shape[0])
    print("Test data size:", X_test.shape[0])

    # 3. 머신러닝 모델 학습
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 4. 예측
    y_pred = model.predict(X_test)

    # 5. 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel: RandomForestClassifier")
    print("Accuracy:", accuracy)
    print("Accuracy (%):", round(accuracy * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))


if __name__ == "__main__":
    main()