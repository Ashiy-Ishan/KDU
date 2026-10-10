import os
import json
import joblib
import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

MODEL_VERSION = "v1.2"
TEMPERATURE = 0.25

CATEGORY_TO_TEAM = {
    "payment_refund": "Payments & Refunds",
    "ride_trip_issue": "Ride Operations",
    "lost_item": "Lost & Found",
    "order_missing_wrong": "Food Operations",
    "delivery_delay": "Delivery Operations",
    "food_quality": "Restaurant Quality",
    "account_promo": "Account Services",
    "safety_conduct": "Trust & Safety",
    "app_technical": "Tech Support",
    "general_inquiry": "Front-line Support",
    "spam_irrelevant": "Auto-close / Spam Filter",
}

def prepare_text(df: pd.DataFrame) -> pd.Series:
    channel = df['channel'].fillna('').astype(str)
    subject = df['subject'].fillna('').astype(str)
    text = df['text'].fillna('').astype(str)
    return channel + " " + subject + " " + text

def train_and_export():
    train_path = "TensorForge 2.0 - MVP Stage Resources/TensorForge 2.0 - MVP Resources DataSet/train.csv"
    val_path = "TensorForge 2.0 - MVP Stage Resources/TensorForge 2.0 - MVP Resources DataSet/validation.csv"
    output_dir = "src/models"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    X_train_raw = prepare_text(train_df)
    X_val_raw = prepare_text(val_df)

    print("Fitting optimized multilingual FeatureUnion vectorizer...")
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True))
    ])
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)
    print(f"Vocabulary dimensions: {X_train.shape[1]} features")

    print("1. Training Primary Category LinearSVC...")
    primary_clf = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    primary_clf.fit(X_train, train_df["category"])
    primary_preds = primary_clf.predict(X_val)
    primary_acc = accuracy_score(val_df["category"], primary_preds)
    primary_f1 = f1_score(val_df["category"], primary_preds, average="macro")
    print(f"   Primary Category Val Accuracy: {primary_acc:.4f}, Macro F1: {primary_f1:.4f}")

    print("2. Training Secondary Issue Detector & Classifier...")
    train_df["has_secondary"] = train_df["secondary_category"].notna() & (train_df["secondary_category"] != "")
    val_df["has_secondary"] = val_df["secondary_category"].notna() & (val_df["secondary_category"] != "")

    sec_detector = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    sec_detector.fit(X_train, train_df["has_secondary"])

    sec_train_subset = train_df[train_df["has_secondary"]]
    X_sec_train = vectorizer.transform(prepare_text(sec_train_subset))
    sec_clf = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    sec_clf.fit(X_sec_train, sec_train_subset["secondary_category"])

    val_has_sec_pred = sec_detector.predict(X_val)
    val_sec_pred = []
    for i, has_s in enumerate(val_has_sec_pred):
        if not has_s:
            val_sec_pred.append(None)
        else:
            p = sec_clf.predict(X_val[i:i+1])[0]
            if p == primary_preds[i]:
                val_sec_pred.append(None)
            else:
                val_sec_pred.append(p)
    
    val_true_sec = [s if (pd.notna(s) and s != "") else None for s in val_df["secondary_category"]]
    val_true_sec_str = [s if s is not None else "NONE" for s in val_true_sec]
    val_pred_sec_str = [s if s is not None else "NONE" for s in val_sec_pred]
    sec_acc = accuracy_score(val_true_sec_str, val_pred_sec_str)
    print(f"   Secondary Category Overall Val Accuracy: {sec_acc:.4f}")

    print("3. Training Urgency Classifier...")
    urgency_clf = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    urgency_clf.fit(X_train, train_df["is_urgent"].astype(bool))
    urgency_preds = urgency_clf.predict(X_val)
    urg_acc = accuracy_score(val_df["is_urgent"].astype(bool), urgency_preds)
    urg_f1 = f1_score(val_df["is_urgent"].astype(bool), urgency_preds, average="macro")
    print(f"   Urgency Val Accuracy: {urg_acc:.4f}, Macro F1: {urg_f1:.4f}")

    artifact_path = os.path.join(output_dir, f"model_{MODEL_VERSION}.joblib")
    artifacts = {
        "model_version": MODEL_VERSION,
        "temperature": TEMPERATURE,
        "vectorizer": vectorizer,
        "primary_clf": primary_clf,
        "sec_detector": sec_detector,
        "sec_clf": sec_clf,
        "urgency_clf": urgency_clf,
        "category_to_team": CATEGORY_TO_TEAM,
        "metrics": {
            "primary_accuracy": float(primary_acc),
            "primary_macro_f1": float(primary_f1),
            "secondary_accuracy": float(sec_acc),
            "urgency_accuracy": float(urg_acc),
            "urgency_macro_f1": float(urg_f1)
        }
    }

    joblib.dump(artifacts, artifact_path, compress=5)
    size_mb = os.path.getsize(artifact_path) / (1024 * 1024)
    print(f"\nModel artifacts successfully saved to: {artifact_path} ({size_mb:.2f} MB)")

    metrics_path = os.path.join(output_dir, "validation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(artifacts["metrics"], f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    train_and_export()
