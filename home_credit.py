import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report


TRAIN_PATH = "data/application_train.csv"
TEST_PATH = "data/application_test.csv"


def main():
    print("=" * 60)
    print("HOME CREDIT DEFAULT RISK")
    print("=" * 60)

    # 1. Load data
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print("Train shape:", train.shape)
    print("Test shape :", test.shape)

    # 2. Separate target
    X = train.drop("TARGET", axis=1)
    y = train["TARGET"]

    print("\nTarget distribution:")
    print(y.value_counts())

    # 3. Identify feature types
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumeric features    :", len(numeric_features))
    print("Categorical features:", len(categorical_features))

    # 4. Preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # 5. Train-validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining rows  :", len(X_train))
    print("Validation rows:", len(X_valid))

    # 6. Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    # 7. Complete pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    # 8. Train
    print("\nTraining Random Forest...")
    pipeline.fit(X_train, y_train)
    print("Training completed!")

    # 9. Validation ROC-AUC
    y_valid_proba = pipeline.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, y_valid_proba)

    print("\nValidation ROC-AUC:", round(roc_auc, 6))

    # 10. Classification report
    y_valid_pred = pipeline.predict(X_valid)
    print("\nClassification Report:")
    print(classification_report(y_valid, y_valid_pred))

    # 11. Predict test data
    print("\nCreating test predictions...")
    test_proba = pipeline.predict_proba(test)[:, 1]

    # 12. Create Kaggle submission
    submission = pd.DataFrame({
        "SK_ID_CURR": test["SK_ID_CURR"],
        "TARGET": test_proba
    })

    submission.to_csv("submission.csv", index=False)

    print("\nSubmission created successfully!")
    print("Submission shape:", submission.shape)
    print(submission.head())
    print("\nFile: submission.csv")


if __name__ == "__main__":
    main()
