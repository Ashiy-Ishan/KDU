"""
Thread-safe, high-performance predictor module for RideEat tickets.
Enforces all business rules and consistency schemas:
  - Fixed category-to-team derivation
  - Spam constraints (secondary=null, is_urgent=false)
  - Non-identical secondary category
  - Confidence calculation and needs_human_review flag
"""

import os
import joblib
import numpy as np

class TicketPredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Default to bundled model in src/models/
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "model_v1.2.joblib")
        
        self.model_path = model_path
        self.is_loaded = False
        self.artifacts = None
        self.vectorizer = None
        self.primary_clf = None
        self.sec_detector = None
        self.sec_clf = None
        self.urgency_clf = None
        self.model_version = None
        self.category_to_team = {}
        self.classes_ = []
        self.human_review_threshold = 0.45

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")
        
        self.artifacts = joblib.load(self.model_path)
        self.model_version = self.artifacts.get("model_version", "v1.2")
        self.vectorizer = self.artifacts["vectorizer"]
        self.primary_clf = self.artifacts["primary_clf"]
        self.sec_detector = self.artifacts["sec_detector"]
        self.sec_clf = self.artifacts["sec_clf"]
        self.urgency_clf = self.artifacts["urgency_clf"]
        self.category_to_team = self.artifacts["category_to_team"]
        self.classes_ = list(self.primary_clf.classes_)
        self.is_loaded = True
        print(f"[TicketPredictor] Successfully loaded model version '{self.model_version}' from {self.model_path}")

    def predict_one(self, channel: str, subject: str, text: str, ticket_id: str = None) -> dict:
        """
        Predicts a single ticket. Returns dictionary conforming to PredictResponse schema.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded yet")

        subject = subject or ""
        text = text or ""
        channel = channel or ""

        full_text = f"{channel} {subject} {text}"
        X = self.vectorizer.transform([full_text])

        # 1. Primary Category & Confidence
        primary_probs = self.primary_clf.predict_proba(X)[0]
        primary_idx = int(np.argmax(primary_probs))
        primary_cat = self.classes_[primary_idx]
        raw_confidence = float(primary_probs[primary_idx])
        # Bound between 0.0 and 1.0, rounded to 4 decimals
        confidence = round(max(0.0, min(1.0, raw_confidence)), 4)

        # 2. Secondary Category
        has_sec = bool(self.sec_detector.predict(X)[0])
        secondary_cat = None
        if has_sec and primary_cat != "spam_irrelevant":
            candidate_sec = str(self.sec_clf.predict(X)[0])
            if candidate_sec != primary_cat:
                secondary_cat = candidate_sec

        # 3. Urgency
        is_urgent_pred = bool(self.urgency_clf.predict(X)[0])

        # 4. Consistency Rules (Enforced by OpenAPI schema & DATA_NOTES.md)
        if primary_cat == "spam_irrelevant":
            secondary_cat = None
            is_urgent_pred = False

        if secondary_cat == primary_cat:
            secondary_cat = None

        team = self.category_to_team.get(primary_cat, "Front-line Support")
        needs_human_review = confidence < self.human_review_threshold

        response = {
            "category": primary_cat,
            "secondary_category": secondary_cat,
            "team": team,
            "is_urgent": is_urgent_pred,
            "confidence": confidence,
            "model_version": self.model_version,
            "needs_human_review": needs_human_review
        }

        if ticket_id is not None:
            response["ticket_id"] = ticket_id

        return response

    def predict_batch(self, tickets: list) -> list:
        """
        Vectorized batch prediction for maximum throughput.
        Each item in tickets is expected to be a dict with: ticket_id, channel, subject, text.
        Returns a list of prediction dicts in identical order.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded yet")

        if not tickets:
            return []

        full_texts = []
        for t in tickets:
            ch = t.get("channel", "") or ""
            subj = t.get("subject", "") or ""
            txt = t.get("text", "") or ""
            full_texts.append(f"{ch} {subj} {txt}")

        X = self.vectorizer.transform(full_texts)

        # Primary
        primary_probs_all = self.primary_clf.predict_proba(X)
        primary_indices = np.argmax(primary_probs_all, axis=1)

        # Secondary
        has_sec_all = self.sec_detector.predict(X)
        sec_candidates = self.sec_clf.predict(X)

        # Urgency
        urgency_all = self.urgency_clf.predict(X)

        results = []
        for i, t in enumerate(tickets):
            prim_idx = int(primary_indices[i])
            prim_cat = self.classes_[prim_idx]
            conf = round(float(primary_probs_all[i, prim_idx]), 4)
            conf = max(0.0, min(1.0, conf))

            sec_cat = None
            if bool(has_sec_all[i]) and prim_cat != "spam_irrelevant":
                cand = str(sec_candidates[i])
                if cand != prim_cat:
                    sec_cat = cand

            is_urg = bool(urgency_all[i])
            if prim_cat == "spam_irrelevant":
                sec_cat = None
                is_urg = False

            if sec_cat == prim_cat:
                sec_cat = None

            team = self.category_to_team.get(prim_cat, "Front-line Support")
            needs_review = conf < self.human_review_threshold

            item_res = {
                "category": prim_cat,
                "secondary_category": sec_cat,
                "team": team,
                "is_urgent": is_urg,
                "confidence": conf,
                "model_version": self.model_version,
                "needs_human_review": needs_review
            }
            if "ticket_id" in t:
                item_res["ticket_id"] = t["ticket_id"]

            results.append(item_res)

        return results

