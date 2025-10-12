"""
ga_feature_selection.py
تنفيذ مُحسّن لاختيار الميزات باستخدام خوارزمية جينية بسيطة مع توازن بين الدقة وعدد الميزات.

الوظائف الرئيسية:
- SimpleGA: تنفيذ GA مع دعم alpha لتقليل عدد الميزات كقيد
- evaluate_features: حساب دقة cross-validated
- baseline_select_kbest, baseline_rfe: طرق تقليدية للمقارنة
- helpers: save/load results
"""

import json
import time
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

# -----------------------
# Utilities
# -----------------------
def _ensure_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """محاولة تحويل الأعمدة النوعية إلى مؤشرات عددية بسيطة (Label Encoding)"""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("category"):
            try:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            except Exception:
                # Last-resort: drop column if cannot be encoded
                df.drop(columns=[col], inplace=True)
    df = df.fillna(df.mean(numeric_only=True))
    return df

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# -----------------------
# Fitness / evaluation
# -----------------------
def evaluate_features(X: pd.DataFrame, y: pd.Series, feature_mask: np.ndarray,
                      estimator=None, cv=5, scoring='accuracy', random_state=42) -> float:
    """
    Returns cross-validated score for estimator using selected features.
    If no feature selected: returns 0.0
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    selected_idx = np.where(feature_mask)[0]
    if len(selected_idx) == 0:
        return 0.0
    X_sel = X.iloc[:, selected_idx]
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    try:
        scores = cross_val_score(estimator, X_sel, y, cv=cv_obj, scoring=scoring, n_jobs=-1)
        return float(np.mean(scores))
    except Exception:
        # If estimator fails (e.g., constant column): penalize
        return 0.0

# -----------------------
# Simple GA implementation
# -----------------------
class SimpleGA:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pop_size: int = 40,
        generations: int = 30,
        cx_prob: float = 0.6,
        mut_prob: float = 0.02,
        tournament_k: int = 3,
        elitism: int = 2,
        estimator=None,
        alpha: float = 0.01,
        cv: int = 5,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        alpha: weight penalizing number of features. Fitness = accuracy - alpha * (n_selected / n_total)
        """
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

        # preprocess X to numeric (label-encode categorical columns)
        self.X_raw = X.copy().reset_index(drop=True)
        self.X = _ensure_numeric_dataframe(self.X_raw)
        if self.X.shape[1] != X.shape[1]:
            # Some columns may have been dropped; keep names aligned
            dropped = set(X.columns) - set(self.X.columns)
            if dropped and verbose:
                print(f"تحذير: تم إسقاط الأعمدة غير القابلة للتحويل: {dropped}")
        self.y = y.reset_index(drop=True)
        self.n_features = self.X.shape[1]
        self.feature_names = self.X.columns.tolist()

        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.cx_prob = float(cx_prob)
        self.mut_prob = float(mut_prob)
        self.tournament_k = int(tournament_k)
        self.elitism = int(elitism)
        self.estimator = estimator if estimator is not None else RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        self.alpha = float(alpha)
        self.cv = int(cv)
        self.verbose = verbose

    def _random_individual(self) -> np.ndarray:
        # randomly include ~30-60% of features
        p = random.uniform(0.25, 0.6)
        return (np.random.rand(self.n_features) < p)

    def _initial_population(self) -> List[np.ndarray]:
        return [self._random_individual() for _ in range(self.pop_size)]

    def _raw_fitness(self, individual: np.ndarray) -> float:
        return evaluate_features(self.X, self.y, individual, estimator=self.estimator, cv=self.cv, random_state=self.random_state)

    def _fitness(self, individual: np.ndarray) -> float:
        """
        Composite fitness: accuracy - alpha * fraction_of_selected_features
        Higher is better.
        """
        acc = self._raw_fitness(individual)
        frac = individual.sum() / max(1, self.n_features)
        return acc - self.alpha * frac

    def _tournament_select(self, population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
        candidates = random.sample(range(len(population)), self.tournament_k)
        best = max(candidates, key=lambda i: fitnesses[i])
        return population[best].copy()

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.cx_prob:
            return a.copy(), b.copy()
        # uniform crossover
        mask = np.random.rand(self.n_features) < 0.5
        child1 = np.where(mask, a, b)
        child2 = np.where(mask, b, a)
        return child1.copy(), child2.copy()

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        for i in range(self.n_features):
            if random.random() < self.mut_prob:
                individual[i] = not individual[i]
        return individual

    def run(self, save_history: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
        pop = self._initial_population()
        # compute fitnesses
        raw_scores = [self._raw_fitness(ind) for ind in pop]
        fitnesses = [raw_scores[i] - self.alpha * (pop[i].sum() / max(1, self.n_features)) for i in range(len(pop))]

        history = []
        start = time.time()
        for gen in range(1, self.generations + 1):
            # elitism: keep top-k
            new_pop = []
            ranked = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)
            for i in range(min(self.elitism, len(ranked))):
                new_pop.append(pop[ranked[i]].copy())

            # produce offspring
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = new_pop
            raw_scores = [self._raw_fitness(ind) for ind in pop]
            fitnesses = [raw_scores[i] - self.alpha * (pop[i].sum() / max(1, self.n_features)) for i in range(len(pop))]

            best_idx = int(np.argmax(fitnesses))
            best_raw = raw_scores[best_idx]
            best_fit = fitnesses[best_idx]
            best_mask = pop[best_idx]
            mean_fit = float(np.mean(fitnesses))
            history.append({
                "generation": gen,
                "best_raw_score": float(best_raw),
                "best_fitness": float(best_fit),
                "mean_fitness": mean_fit,
                "n_features": int(best_mask.sum())
            })

            if self.verbose:
                print(f"[Gen {gen}/{self.generations}] best_raw: {best_raw:.4f} best_fit: {best_fit:.4f} features: {int(best_mask.sum())}")

        end = time.time()
        elapsed = end - start

        # final result (re-evaluate best on raw score)
        final_raw_scores = [self._raw_fitness(ind) for ind in pop]
        best_idx = int(np.argmax(final_raw_scores))
        best_mask = pop[best_idx]
        best_raw = final_raw_scores[best_idx]
        selected_features = [self.feature_names[i] for i in range(self.n_features) if best_mask[i]]

        result = {
            "best_mask": best_mask.tolist(),
            "best_raw_score": float(best_raw),
            "selected_features": selected_features,
            "n_selected": int(best_mask.sum()),
            "history": history,
            "elapsed_time_sec": elapsed,
            "n_total_features": self.n_features,
            "alpha": self.alpha,
            "params": {
                "pop_size": self.pop_size,
                "generations": self.generations,
                "cx_prob": self.cx_prob,
                "mut_prob": self.mut_prob,
                "tournament_k": self.tournament_k,
                "elitism": self.elitism,
                "cv": self.cv
            }
        }

        if save_history and save_path:
            try:
                save_json(save_path, result)
            except Exception:
                pass

        return result

# -----------------------
# Baseline methods
# -----------------------
def baseline_select_kbest(X: pd.DataFrame, y: pd.Series, k: int = 10, estimator=None, cv: int = 5) -> Dict[str, Any]:
    k = max(1, min(k, X.shape[1]))
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    score = evaluate_features(X, y, mask, estimator=estimator, cv=cv)
    return {"mask": mask, "score": float(score), "feature_names": X.columns[mask].tolist()}

def baseline_rfe(X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 10, estimator=None, cv: int = 5) -> Dict[str, Any]:
    n_features_to_select = max(1, min(n_features_to_select, X.shape[1]))
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=0.1)
    selector = selector.fit(X, y)
    mask = selector.support_
    score = evaluate_features(X, y, mask, estimator=estimator, cv=cv)
    return {"mask": mask, "score": float(score), "feature_names": X.columns[mask].tolist()}
