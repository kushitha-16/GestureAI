import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_FILE = "gesture_data.csv"
MODEL_FILE = "gesture_model.pkl"

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully")
print("Total samples:", len(df))
print("\nSamples per gesture:")
print(df["label"].value_counts())

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_FILE)

print(f"\nModel saved as {MODEL_FILE}")