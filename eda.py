import pandas as pd

TRAIN_PATH = "data/application_train.csv"

train = pd.read_csv(TRAIN_PATH)

print("Dataset shape:", train.shape)
print("\nColumns:")
print(train.columns.tolist())

print("\nFirst 5 rows:")
print(train.head())

print("\nMissing values - top 20:")
missing = train.isnull().sum().sort_values(ascending=False)
print(missing.head(20))

print("\nTarget distribution:")
print(train["TARGET"].value_counts())

print("\nTarget percentage:")
print(train["TARGET"].value_counts(normalize=True) * 100)
