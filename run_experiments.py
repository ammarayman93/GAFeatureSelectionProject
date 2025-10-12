"""
run_experiments.py
سكربت تشغيل تلقائي لمجموعة بيانات تجريبية (breast_cancer) ويحتفظ بالنتائج JSON/CSV.
"""

import os
import json
from datetime import datetime

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from ga_feature_selection import SimpleGA, baseline_select_kbest, baseline_rfe, save_json

OUTPUT_DIR = "results"

def load_example():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    ensure_dir(OUTPUT_DIR)
    X, y = load_example()
    estimator = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)

    print("Running GA feature selection (example dataset)...")
    ga = SimpleGA(X, y, pop_size=40, generations=20, cx_prob=0.7, mut_prob=0.03, elitism=2, estimator=estimator, alpha=0.01, cv=5, random_state=42, verbose=True)
    res = ga.run(save_history=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(OUTPUT_DIR, f"ga_result_{timestamp}.json")
    save_json(outpath, res)
    print(f"Saved GA result to: {outpath}")

    print("\nBaseline SelectKBest (k=10):")
    kbest = baseline_select_kbest(X, y, k=10, estimator=estimator, cv=5)
    print(kbest["score"], kbest["feature_names"])

    print("\nBaseline RFE (n=10):")
    rfe = baseline_rfe(X, y, n_features_to_select=10, estimator=estimator, cv=5)
    print(rfe["score"], rfe["feature_names"])

    # save summary CSV
    summary = {
        "dataset": "breast_cancer",
        "ga_best_score": res["best_raw_score"],
        "ga_n_selected": res["n_selected"],
        "kbest_score": kbest["score"],
        "kbest_n_selected": len(kbest["feature_names"]),
        "rfe_score": rfe["score"],
        "rfe_n_selected": len(rfe["feature_names"])
    }
    summary_path = os.path.join(OUTPUT_DIR, f"summary_{timestamp}.json")
    save_json(summary_path, summary)
    print(f"Saved summary to: {summary_path}")

if __name__ == "__main__":
    main()
