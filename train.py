import json
import joblib
import pandas as pd
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold,
                                     cross_val_score, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from features import make_features, load_data

data = load_data()
X = make_features(data)
y = data["win"]

# ---------- 3.1: Chia dữ liệu, train model đầu tiên ----------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(x_train.shape, x_test.shape)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=0.1, max_iter=2000)),
])
model.fit(x_train, y_train)

print("Train:", accuracy_score(y_train, model.predict(x_train)))
print("Test:", accuracy_score(y_test, model.predict(x_test)))

# ---------- 3.2: Cross-validation ----------
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print("So diem:", len(scores))
print("Accuracy trung binh:", scores.mean())
print("Do lech:", scores.std())
print("Thap nhat:", scores.min(), "| Cao nhat:", scores.max())


# ---------- 3.3: So sánh 4 thuật toán (lưu kết quả vào rows) ----------
def make_pipe(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


models = {
    "Logistic Regression": make_pipe(LogisticRegression(C=0.1, max_iter=2000)),
    "SVC": make_pipe(SVC(C=1, probability=True, random_state=0)),
    "Gradient Boosting": make_pipe(GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0)),
    "Random Forest": make_pipe(RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, random_state=0)),
}

rows = []
print("\nSo sanh model:")
for name, m in models.items():
    r = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    acc = r["test_accuracy"]
    print(f"{name:20s} {acc.mean():.3f} ± {acc.std():.3f}")
    rows.append({"Model": name, "Accuracy": acc.mean(), "Accuracy_std": acc.std(),
                 "F1": r["test_f1"].mean(), "ROC-AUC": r["test_roc_auc"].mean()})

# ---------- 3.4: Ensemble ----------
ensemble = VotingClassifier([
    ("lr", LogisticRegression(C=0.1, max_iter=2000)),
    ("svc", SVC(C=1, probability=True, random_state=0)),
    ("gb", GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0)),
    ("rf", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, random_state=0)),
], voting="soft")
ens_pipe = make_pipe(ensemble)

r = cross_validate(ens_pipe, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
print(f"\nEnsemble accuracy: {r['test_accuracy'].mean():.3f} ± {r['test_accuracy'].std():.3f}")
print(f"Ensemble ROC-AUC: {r['test_roc_auc'].mean():.3f}")
rows.append({"Model": "Ensemble (voting)",
             "Accuracy": r["test_accuracy"].mean(), "Accuracy_std": r["test_accuracy"].std(),
             "F1": r["test_f1"].mean(), "ROC-AUC": r["test_roc_auc"].mean()})

# Chấm riêng ensemble trên tập test 20% (chưa từng thấy lúc học)
ens_pipe.fit(x_train, y_train)
p = ens_pipe.predict_proba(x_test)[:, 1]
pred = (p >= 0.5).astype(int)
metrics = {"holdout_accuracy": accuracy_score(y_test, pred),
           "holdout_f1": f1_score(y_test, pred),
           "holdout_auc": roc_auc_score(y_test, p),
           "n_train": len(x_train), "n_test": len(x_test)}
print("Hold-out:", metrics)

# Lưu kết quả để tab "Về mô hình" của app đọc lại
pd.DataFrame(rows).to_csv("model_comparison.csv", index=False)
json.dump(metrics, open("metrics.json", "w"), indent=2)

# Train lại trên TOÀN BỘ data để dùng thật, rồi lưu
ens_pipe.fit(X, y)
explainer = make_pipe(LogisticRegression(C=0.1, max_iter=2000)).fit(X, y)
joblib.dump({"model": ens_pipe, "explainer": explainer,
             "features": list(X.columns)}, "model.pkl")

# Kiểm tra: nạp lại model và dự đoán 3 trận đầu
m = joblib.load("model.pkl")["model"]
print("Xac suat thang 3 tran dau:", m.predict_proba(X.head(3))[:, 1].round(3))
print("Ket qua that:", y.head(3).tolist())