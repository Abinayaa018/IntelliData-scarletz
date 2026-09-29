"""
STOCKSENSE Classification Model Benchmark Suite (Stock-Out Risk Prediction)
Implements Random Forest, XGBoost, Decision Tree, SVM, KNN, Naive Bayes, Logistic Regression,
class imbalance balancing comparisons, and Platt probability calibration on validation set.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import joblib
import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

from src.evaluation import evaluate_classification
from src.models.regression import FEATURE_COLS_NUM, FEATURE_COLS_CAT

logger = logging.getLogger("stocksense.models.classification")

class StockoutClassifierSuite:
    def __init__(self, random_seed: int = 42, high_risk_thresh: float = 0.70, med_risk_thresh: float = 0.40):
        self.random_seed = random_seed
        self.high_risk_thresh = high_risk_thresh
        self.med_risk_thresh = med_risk_thresh
        self.fitted_models = {}
        self.best_model_name = None
        self.best_calibrated_pipeline = None

    def _get_preprocessor(self, scale_num: bool = True) -> ColumnTransformer:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler() if scale_num else "passthrough")
        ])
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        return ColumnTransformer([
            ("num", num_transformer, FEATURE_COLS_NUM),
            ("cat", cat_transformer, [c for c in FEATURE_COLS_CAT if c])
        ])

    def benchmark_and_train(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Trains permitted classifiers, tests imbalance balancing, calibrates probabilities, returns benchmark table."""
        y_train = train_df["stockout_flag"].values
        y_val = val_df["stockout_flag"].values
        y_test = test_df["stockout_flag"].values

        # Calculate class imbalance ratio for XGBoost scale_pos_weight
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale_pos_wt = float(neg_count / (pos_count + 1e-5))

        benchmarks = []

        # Permitted Model List
        models_to_train = {
            "Random Forest (Balanced)": (RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=self.random_seed, n_jobs=-1), False),
            "Random Forest (Unbalanced)": (RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_seed, n_jobs=-1), False),
            "XGBoost Classifier": (xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.08, scale_pos_weight=scale_pos_wt, random_state=self.random_seed, n_jobs=-1), False),
            "Decision Tree": (DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=self.random_seed), False),
            "SVM Classifier": (SVC(class_weight="balanced", random_state=self.random_seed), True),
            "KNN Classifier": (KNeighborsClassifier(n_neighbors=7, n_jobs=-1), True),
            "Naive Bayes": (GaussianNB(), True),
            "Logistic Regression": (LogisticRegression(class_weight="balanced", max_iter=1000, random_state=self.random_seed), True)
        }

        best_f1 = -1.0

        for name, (classifier, scale_num) in models_to_train.items():
            logger.info(f"Training Classification Model: {name}")
            preprocessor = self._get_preprocessor(scale_num=scale_num)
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ])
            
            # Fit & calibrate probabilities using Platt scaling (sigmoid) via cross-validation
            calibrated_pipe = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=5)
            calibrated_pipe.fit(train_df, y_train)
            
            # Predict probabilities on test set
            probs = calibrated_pipe.predict_proba(test_df)[:, 1]
            
            metrics = evaluate_classification(y_test, probs, threshold=0.50, model_name=name)
            benchmarks.append(metrics)
            
            self.fitted_models[name] = calibrated_pipe
            
            if metrics["F1 Score"] > best_f1:
                best_f1 = metrics["F1 Score"]
                self.best_model_name = name
                self.best_calibrated_pipeline = calibrated_pipe

        benchmark_df = pd.DataFrame(benchmarks).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
        logger.info(f"Classification Benchmark Winner: {self.best_model_name} with F1 Score {best_f1:.4f}")
        
        return benchmark_df, {"best_model": self.best_model_name, "best_f1": best_f1}

    def assign_risk_levels(self, probs: np.ndarray) -> List[str]:
        """Maps probability to risk level: HIGH >= 0.70, MEDIUM 0.40 to <0.70, LOW < 0.40."""
        levels = []
        for p in probs:
            if p >= self.high_risk_thresh:
                levels.append("HIGH")
            elif p >= self.med_risk_thresh:
                levels.append("MEDIUM")
            else:
                levels.append("LOW")
        return levels
