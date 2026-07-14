# ── Import libraries ─────────────────────────────────────────
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# ── Load dataset ─────────────────────────────────────────────
df = pd.read_csv("Crop_recommendation.csv")

print("Dataset loaded successfully")
print(df.head())


# ── Check and handle missing values ──────────────────────────
print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()


# ── Features and target ──────────────────────────────────────
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']


# ── Encode crop labels ───────────────────────────────────────
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# ── Scale features ───────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── Train-test split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# ── Train multiple models ────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_model_name = ""
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{name} Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name


# ── Show best model result ───────────────────────────────────
print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)

y_pred_best = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_best,
    target_names=label_encoder.classes_
))


# ── Save model files ─────────────────────────────────────────
os.makedirs("model", exist_ok=True)

with open("model/crop_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
